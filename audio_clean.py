"""Tratamento do áudio da gravação (etapa "Tratamento do áudio" do Moovi
Marketing). Só ffmpeg, nada sai do servidor. A cadeia se monta a partir de uma
medição do áudio, e cada filtro só entra se a medição pedir:

1. Canal mudo: se um dos dois canais vier vazio, o canal bom é copiado para
   os dois (pan).
2. Declip (adeclip): só se o áudio estiver estourado.
3. Passa-alta em 80 Hz: tira ronco e grave de mesa.
4. Redução de ruído (afftdn, acompanha o ruído de fundo ao longo do vídeo):
   forte (nr 18) com SNR abaixo de 20 dB, leve (nr 10) no resto, e não entra
   se o fundo já estiver limpo (SNR de 35 dB para cima). O time pode forçar
   leve, forte ou desligado.
5. Compressor suave 2:1: só se a voz oscilar muito (LRA acima de 8 LU).
6. loudnorm em 2 passadas: -14 LUFS, pico máximo de -1,5 dBTP (o padrão de
   YouTube, Instagram e LinkedIn).

O resultado vai por PUT direto no MinIO, nas URLs assinadas de quem pediu: o
áudio tratado (AAC em .m4a, mesmo relógio do vídeo de origem) e, se pedido, o
próprio vídeo com o áudio trocado (imagem copiada sem reencode, MP4). Esse
vídeo vira o vídeo do post: transcrição, cortes, edição e render leem ele. A
resposta traz a medição e o que foi aplicado, para a tela explicar.

As funções de leitura da saída do ffmpeg (parse_*) e o plano da cadeia
(plan_chain) não dependem de ffmpeg: dá para testar com Python puro.
"""

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable

import requests

log = logging.getLogger("speech")

TARGET_I = -14.0
TARGET_TP = -1.5
TARGET_LRA = 11.0
HIGHPASS_HZ = 80

# Relação voz/ruído (dB): abaixo disso o ruído é forte; daqui para cima o
# fundo já está limpo e a redução de ruído não entra.
SNR_STRONG_BELOW = 20.0
SNR_CLEAN_FROM = 35.0
NR_STRONG = 18
NR_LIGHT = 10

# Variação de volume da voz (LRA, em LU) acima da qual entra o compressor.
LRA_COMPRESS_ABOVE = 8.0

# Canal mudo: RMS abaixo disso, ou tanto abaixo do outro canal.
DEAD_CHANNEL_DB = -70.0
DEAD_CHANNEL_GAP_DB = 40.0

# Estouro: pico no teto e muitas amostras grudadas nele.
CLIP_PEAK_DB = -0.1
CLIP_MIN_PEAKS = 50

# Janela da medição de voz e ruído (percentis do volume por janela).
WINDOW_SECONDS = 0.05
WINDOW_RATE = 16000

NOISE_MODES = ("auto", "light", "strong", "off")


def _float(text: str | None) -> float:
    if text is None:
        return float("nan")
    t = text.strip().lower()
    if t in ("-inf", "-infinity"):
        return float("-inf")
    if t in ("inf", "+inf", "infinity"):
        return float("inf")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _finite(value: float | None) -> float | None:
    return round(value, 1) if value is not None and math.isfinite(value) else None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --- Leitura da saída do ffmpeg -----------------------------------------------


def parse_channel_stats(stderr: str) -> tuple[list[dict], dict]:
    """Resumo do `astats` (por canal e geral): {"RMS level dB": "-23.4", ...}."""
    channels: list[dict] = []
    overall: dict = {}
    current: dict | None = None
    for line in stderr.splitlines():
        if "astats" not in line or "]" not in line:
            continue
        body = line.split("]", 1)[1].strip()
        m = re.match(r"Channel:\s*(\d+)", body)
        if m:
            current = {}
            channels.append(current)
            continue
        if body.startswith("Overall"):
            current = overall
            continue
        if current is None:
            continue
        key, sep, value = body.partition(":")
        if sep:
            current[key.strip()] = value.strip()
    return channels, overall


_WINDOW_RMS = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(\S+)")


def parse_window_levels(stderr: str) -> list[float]:
    """Volume (RMS, dBFS) de cada janela curta, na ordem do áudio."""
    return [_float(m.group(1)) for m in _WINDOW_RMS.finditer(stderr)]


def speech_and_floor(levels: list[float]) -> tuple[float, float] | None:
    """Nível da voz (percentil 95) e do ruído de fundo (percentil 10) das
    janelas. Silêncio digital (abaixo de -90 dB) não conta: é corte, não ruído."""
    values = sorted(v for v in levels if math.isfinite(v) and v > -90)
    if len(values) < 20:
        return None
    floor = values[int(0.10 * (len(values) - 1))]
    speech = values[int(0.95 * (len(values) - 1))]
    return speech, floor


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def parse_loudnorm(stderr: str) -> dict[str, float] | None:
    """Medição que o loudnorm imprime (print_format=json), em números."""
    found = _LOUDNORM_JSON.findall(stderr)
    if not found:
        return None
    try:
        raw = json.loads(found[-1])
    except ValueError:
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key == "normalization_type":
            continue
        out[key] = _float(str(value))
    out["linear"] = 1.0 if str(raw.get("normalization_type", "")).lower() == "linear" else 0.0
    return out


# --- Medição e plano ----------------------------------------------------------


def summarize(
    channels: list[dict],
    levels: list[float],
    loudness: dict[str, float] | None,
) -> dict:
    """Medição do áudio a partir das três leituras do ffmpeg."""
    rms = [_float(c.get("RMS level dB")) for c in channels]
    dead: int | None = None
    if len(rms) == 2:
        lo = 0 if rms[0] < rms[1] else 1
        hi = 1 - lo
        if math.isfinite(rms[hi]) and (
            not math.isfinite(rms[lo])
            or rms[lo] < DEAD_CHANNEL_DB
            or rms[hi] - rms[lo] > DEAD_CHANNEL_GAP_DB
        ):
            dead = lo
    peaks_db = [_float(c.get("Peak level dB")) for c in channels]
    counts = [_float(c.get("Peak count")) for c in channels]
    peak_counts = [int(v) if math.isfinite(v) else 0 for v in counts]
    # O canal mudo não conta para estouro.
    live = [i for i in range(len(channels)) if i != dead]
    peak = max((peaks_db[i] for i in live if math.isfinite(peaks_db[i])), default=float("-inf"))
    clipped = any(
        math.isfinite(peaks_db[i]) and peaks_db[i] >= CLIP_PEAK_DB and peak_counts[i] >= CLIP_MIN_PEAKS
        for i in live
    )
    sf = speech_and_floor(levels)
    speech, floor = sf if sf else (None, None)
    snr = speech - floor if sf else None
    loud = loudness or {}
    return {
        "channels": len(channels),
        "dead_channel": dead,
        "clipped": clipped,
        "peak_db": _finite(peak),
        "speech_db": _finite(speech),
        "floor_db": _finite(floor),
        "snr": _finite(snr),
        "input_i": _finite(loud.get("input_i")),
        "input_tp": _finite(loud.get("input_tp")),
        "input_lra": _finite(loud.get("input_lra")),
    }


def plan_chain(analysis: dict, noise_mode: str = "auto") -> tuple[list[str], list[dict]]:
    """Filtros (sem o loudnorm, que precisa da medição da cadeia) e, para a
    tela, cada etapa considerada: se entrou e por quê."""
    chain: list[str] = []
    steps: list[dict] = []

    dead = analysis.get("dead_channel")
    if dead is not None:
        good = 1 - dead
        chain.append(f"pan=stereo|c0=c{good}|c1=c{good}")
        steps.append({"key": "channel", "applied": True, "dead": dead})
    else:
        steps.append({"key": "channel", "applied": False})

    if analysis.get("clipped"):
        chain.append("adeclip")
        steps.append({"key": "declip", "applied": True})
    else:
        steps.append({"key": "declip", "applied": False})

    chain.append(f"highpass=f={HIGHPASS_HZ}")
    steps.append({"key": "highpass", "applied": True})

    snr = analysis.get("snr")
    if noise_mode == "off":
        level, reason = None, "off"
    elif noise_mode in ("light", "strong"):
        level, reason = noise_mode, "forced"
    elif snr is None:
        level, reason = "light", "unmeasured"
    elif snr >= SNR_CLEAN_FROM:
        level, reason = None, "clean"
    elif snr < SNR_STRONG_BELOW:
        level, reason = "strong", "auto"
    else:
        level, reason = "light", "auto"
    if level:
        nr = NR_STRONG if level == "strong" else NR_LIGHT
        floor = analysis.get("floor_db")
        nf = _clamp(floor if floor is not None else -50.0, -80.0, -20.0)
        chain.append(f"afftdn=nr={nr}:nf={nf:.0f}:tn=1")
    steps.append({"key": "denoise", "applied": bool(level), "level": level, "reason": reason})

    lra = analysis.get("input_lra")
    if lra is not None and lra > LRA_COMPRESS_ABOVE:
        base = analysis.get("input_i")
        threshold_db = _clamp((base if base is not None else -24.0) + 2.0, -40.0, -6.0)
        threshold = 10 ** (threshold_db / 20)
        chain.append(
            f"acompressor=threshold={threshold:.5f}:ratio=2:attack=15:release=200:knee=4"
        )
        steps.append({"key": "compressor", "applied": True})
    else:
        steps.append({"key": "compressor", "applied": False})

    steps.append({"key": "loudnorm", "applied": True})
    return chain, steps


def _loudnorm(measured: dict[str, float] | None = None) -> str:
    base = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json"
    if not measured:
        return base
    return (
        f"{base}:measured_I={measured['input_i']:.2f}:measured_TP={measured['input_tp']:.2f}"
        f":measured_LRA={measured['input_lra']:.2f}:measured_thresh={measured['input_thresh']:.2f}"
        f":offset={measured['target_offset']:.2f}:linear=true"
    )


# --- Execução -----------------------------------------------------------------


def _ffmpeg(args: list[str]) -> str:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-y", *args], capture_output=True)
    stderr = proc.stderr.decode("utf-8", "ignore")
    if proc.returncode != 0:
        if "does not contain any stream" in stderr or "matches no streams" in stderr:
            raise RuntimeError("A gravação não tem trilha de áudio.")
        log.error("ffmpeg (tratamento do áudio) saiu %s. stderr:\n%s", proc.returncode, stderr[-4000:])
        raise RuntimeError(f"ffmpeg falhou: {stderr[-400:]}")
    return stderr


def measure(path: str) -> dict:
    """Três passadas rápidas: canais/picos, volume por janela e loudness."""
    channels, _ = parse_channel_stats(_ffmpeg(["-i", path, "-af", "astats", "-f", "null", "-"]))
    window = int(WINDOW_RATE * WINDOW_SECONDS)
    levels = parse_window_levels(
        _ffmpeg(
            [
                "-i", path, "-af",
                f"aformat=sample_fmts=flt:sample_rates={WINDOW_RATE}:channel_layouts=mono,"
                f"asetnsamples=n={window}:p=0,astats=metadata=1:reset=1,"
                "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
                "-f", "null", "-",
            ]
        )
    )
    loudness = parse_loudnorm(_ffmpeg(["-i", path, "-af", _loudnorm(), "-f", "null", "-"]))
    return summarize(channels, levels, loudness)


def clean(
    url: str,
    out_path: str,
    noise_mode: str = "auto",
    set_phase: Callable[[str], None] = lambda _p: None,
    video_out: str | None = None,
) -> dict:
    """Extrai o áudio da URL, mede, trata e grava o .m4a em `out_path`. Com
    `video_out`, grava também o vídeo da URL com o áudio tratado no lugar."""
    if noise_mode not in NOISE_MODES:
        noise_mode = "auto"
    workdir = tempfile.mkdtemp(prefix="audio-clean-")
    try:
        src = os.path.join(workdir, "src.flac")
        picture = os.path.join(workdir, "picture.mp4")
        set_phase("audio")
        # Uma leitura só da URL: o áudio para tratar e (se pedido) a imagem
        # copiada como está, para montar o vídeo no fim.
        _ffmpeg(
            [
                "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                "-i", url,
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "flac", "-ar", "48000", src,
                *(
                    ["-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy", picture]
                    if video_out
                    else []
                ),
            ]
        )

        set_phase("measuring")
        analysis = measure(src)
        chain, steps = plan_chain(analysis, noise_mode)

        set_phase("cleaning")
        first = parse_loudnorm(
            _ffmpeg(["-i", src, "-af", ",".join([*chain, _loudnorm()]), "-f", "null", "-"])
        )
        needed = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        if not first or not all(math.isfinite(first.get(k, float("nan"))) for k in needed):
            raise RuntimeError("O áudio está mudo: não há o que tratar.")
        final = _ffmpeg(
            [
                "-i", src, "-af", ",".join([*chain, _loudnorm(first), "aresample=48000"]),
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
                out_path,
            ]
        )
        after = parse_loudnorm(final) or {}

        if video_out:
            set_phase("video")
            _ffmpeg(
                [
                    "-i", picture, "-i", out_path,
                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-movflags", "+faststart",
                    video_out,
                ]
            )
        return {
            "noise_mode": noise_mode,
            "analysis": analysis,
            "steps": steps,
            "output": {
                "i": _finite(after.get("output_i")),
                "tp": _finite(after.get("output_tp")),
                "lra": _finite(after.get("output_lra")),
                "linear": bool(after.get("linear")),
                "size": os.path.getsize(out_path),
            },
            **({"video": {"size": os.path.getsize(video_out)}} if video_out else {}),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _put(url: str, path: str, content_type: str, what: str, tag: str) -> None:
    with open(path, "rb") as f:  # streaming (não carrega tudo na RAM)
        put = requests.put(url, data=f, headers={"Content-Type": content_type}, timeout=3600)
    if put.status_code not in (200, 204):
        log.error("%s | PUT do %s falhou (%s): %s", tag, what, put.status_code, put.text[:500])
        raise RuntimeError(f"Não foi possível gravar o {what} (PUT {put.status_code}).")


def run(params: dict, set_phase: Callable[[str], None]) -> dict:
    """Job da fila (url_jobs): trata e faz o PUT do .m4a (e do vídeo com o
    áudio tratado, se veio `video_upload_url`) nas URLs assinadas."""
    tag = f"audio-job={params['job_id']}"
    started = time.monotonic()
    want_video = bool(params.get("video_upload_url"))
    log.info("%s | INICIADO (ruído=%s, vídeo=%s)", tag, params.get("noise", "auto"), want_video)
    fd, out_path = tempfile.mkstemp(suffix=".m4a")
    os.close(fd)
    video_path: str | None = None
    if want_video:
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
    try:
        result = clean(params["url"], out_path, params.get("noise", "auto"), set_phase, video_path)
        set_phase("uploading")
        _put(params["upload_url"], out_path, "audio/mp4", "áudio tratado", tag)
        if video_path:
            _put(params["video_upload_url"], video_path, "video/mp4", "vídeo com o áudio tratado", tag)
        steps = ", ".join(s["key"] for s in result["steps"] if s["applied"])
        log.info(
            "%s | OK em %.1fs (snr=%s, lra=%s, filtros: %s)",
            tag, time.monotonic() - started, result["analysis"]["snr"],
            result["analysis"]["input_lra"], steps,
        )
        return result
    except Exception as e:  # noqa: BLE001
        log.error("%s | FALHOU em %.1fs: %s", tag, time.monotonic() - started, e)
        raise
    finally:
        for path in (out_path, video_path):
            if path and os.path.exists(path):
                os.remove(path)
