"""Montagem final do vídeo do editor de gravações, com ffmpeg.

POR QUE AQUI: até agora a montagem rodava no navegador do professor (WebCodecs).
Isso amarrava o resultado à máquina dele — memória (o export segurava vários GB
de uma vez) e codec (o encoder de software do Chrome só faz H.264 Baseline, então
quem não tinha encoder de hardware simplesmente não conseguia exportar). Aqui o
ambiente é um só, conhecido e igual pra todo mundo, e o professor pode fechar a
aba no meio.

O DESENHO NÃO É REIMPLEMENTADO AQUI. Fundo (gradiente), sombras, cantos
arredondados, borda da câmera e logo continuam sendo desenhados pelo compositor
do Class — o MESMO código que pinta o preview. O que chega aqui já são PNGs
prontos ("under" = o que fica atrás do vídeo, "over" = o que fica na frente) e as
máscaras de recorte. Este módulo só faz o que o ffmpeg faz melhor: cortar,
escalar, sobrepor, acelerar, misturar áudio e codificar. É isso que garante o MP4
igual ao preview sem manter dois compositores vivos.

ESTRATÉGIA: um ffmpeg POR SEGMENTO (clip), todos com parâmetros de encode
idênticos, e no fim `concat` com `-c copy`. Um único filter_complex com
`split`+`trim`+`concat` pra timeline inteira seria mais elegante e é uma
armadilha conhecida: o `concat` consome os ramos em ordem, então o ffmpeg
bufferiza na RAM todos os frames dos ramos seguintes — trocaríamos o estouro de
memória do navegador por um estouro de memória no servidor.
"""

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse
from datetime import datetime, timezone

import requests
from pydantic import BaseModel, Field

import db

log = logging.getLogger("speech")

# Renders simultâneos. Um render come uma CPU inteira por minutos; sem teto, cinco
# professores exportando ao mesmo tempo derrubam o serviço inteiro (inclusive a
# transcrição, que divide o mesmo processo). O resto espera na fila.
RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", "2"))
# Teto da fila: cheia, o pedido é RECUSADO em vez de aceitar um trabalho que só
# começaria daqui a uma hora — o professor precisa saber disso na hora.
RENDER_QUEUE_MAX = int(os.getenv("RENDER_QUEUE_MAX", "32"))

# Qualidade do MP4 final. CRF 20 dá um 1080p de aula visualmente transparente e
# MUITO menor que o VBR de 8 Mbps que o export do navegador usava.
FINAL_CRF = int(os.getenv("RENDER_CRF", "20"))
FINAL_PRESET = os.getenv("RENDER_PRESET", "veryfast")
AUDIO_BITRATE = "128k"
SAMPLE_RATE = 48_000

# Peso de cada fase na barra de progresso (soma 1). O vídeo domina porque é o
# único passo que re-codifica frame a frame.
W_DOWNLOAD = 0.08
W_VIDEO = 0.72
W_AUDIO = 0.10
W_MUX = 0.05
W_UPLOAD = 0.05


# --- Contrato com o moovi_class -------------------------------------------
# O Class é quem calcula a geometria (ele tem o compositor). Tudo chega aqui já
# em PIXELS do quadro de saída — este módulo NÃO decide layout.


class Rect(BaseModel):
    x: float
    y: float
    w: float
    h: float


class VideoLayer(BaseModel):
    """Uma fonte de vídeo desenhada no segmento (a tela/asset ou a câmera)."""

    source: str
    """Chave em `RenderPlan.sources`."""
    start: float = 0.0
    """Tempo na FONTE onde o segmento começa (segundos)."""
    rect: Rect
    """Onde desenhar, em pixels do quadro de saída."""
    crop: Rect | None = None
    """Recorte na fonte, NORMALIZADO (0..1). É o `screenCrop` do editor."""
    fit: str = "contain"
    """`contain`: o rect já respeita a proporção da fonte (tela/asset), então é só
    escalar. `cover`: preenche o rect recortando o excedente pelo centro (câmera)."""
    mask_url: str | None = None
    """PNG cinza do tamanho do rect = o alpha do recorte (cantos arredondados,
    círculo). Ausente = retângulo cheio, e aí nem entra no grafo."""
    mirror: bool = False
    brightness: float = 1.0
    contrast: float = 1.0


class Segment(BaseModel):
    """Um clip da timeline já resolvido: quanto dura, o que desenha, como abre e fecha."""

    dur: float
    """Duração na TIMELINE (já dividida pela velocidade)."""
    speed: float = 1.0
    under_url: str
    """PNG RGBA do tamanho do quadro: fundo e sombras — tudo que fica ATRÁS."""
    over_url: str | None = None
    """PNG RGBA do tamanho do quadro: borda da câmera e logo — o que fica NA FRENTE."""
    layers: list[VideoLayer] = Field(default_factory=list)
    fade_in: float = 0.0
    fade_out: float = 0.0


class AudioClip(BaseModel):
    """Áudio que acompanha um clip: segue corte e velocidade (o tom sobe/desce)."""

    source: str
    start: float
    dur: float
    """Duração na TIMELINE."""
    at: float
    """Onde entra na timeline."""
    speed: float = 1.0
    gain: float = 1.0


class AudioBlock(BaseModel):
    """Bloco de faixa paralela (música, vinheta): tempo de TIMELINE, sem velocidade."""

    source: str
    at: float
    dur: float
    offset: float = 0.0
    gain: float = 0.15
    fade_in: float = 0.0
    fade_out: float = 0.0
    loop: bool = True


class RenderPlan(BaseModel):
    recording_id: str
    spec_hash: str
    upload_url: str
    """Presigned PUT do `export.mp4` no MinIO."""
    sources: dict[str, str]
    """Chave lógica → URL de leitura. Cada uma é baixada UMA vez."""
    duration: float
    width: int = 1920
    height: int = 1080
    fps: int = 30
    segments: list[Segment] = Field(default_factory=list)
    audio_clips: list[AudioClip] = Field(default_factory=list)
    audio_blocks: list[AudioBlock] = Field(default_factory=list)
    limiter: bool = True

    def urls(self) -> list[str]:
        """Toda URL do plano, pro chamador validar contra SSRF de uma vez só."""
        out = [self.upload_url, *self.sources.values()]
        for s in self.segments:
            out.append(s.under_url)
            if s.over_url:
                out.append(s.over_url)
            for lay in s.layers:
                if lay.mask_url:
                    out.append(lay.mask_url)
        return out


class RenderCancelled(Exception):
    """A gravação saiu de QUEUED/RENDERING — cancelaram ou pediram outro export."""


# --- Encoder ---------------------------------------------------------------


_encoder_cache: str | None = None
_encoder_lock = threading.Lock()


def video_encoder() -> str:
    """`h264_nvenc` quando a máquina tem GPU NVIDIA utilizável; senão `libx264`.

    Detectado UMA vez e cacheado. O nvenc roda 1080p muito acima do tempo real e
    é o que transforma um render de minutos num de segundos; sem GPU o x264
    entrega o mesmo arquivo, só mais devagar.
    """
    global _encoder_cache
    with _encoder_lock:
        if _encoder_cache is not None:
            return _encoder_cache
        forced = os.getenv("RENDER_ENCODER", "").strip()
        if forced:
            _encoder_cache = forced
            log.info("render | encoder forçado por env: %s", forced)
            return forced
        _encoder_cache = "libx264"
        try:
            listed = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            if "h264_nvenc" in listed:
                # Listar não é poder usar: sem driver ou sem GPU visível no
                # container o nvenc só falha na hora do encode, no meio do
                # trabalho do professor. Um teste de 1 frame decide agora.
                probe = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                     "-c:v", "h264_nvenc", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=60,
                )
                if probe.returncode == 0:
                    _encoder_cache = "h264_nvenc"
        except Exception as e:  # noqa: BLE001
            log.warning("render | detecção de encoder falhou (%s); usando libx264", e)
        log.info("render | encoder de vídeo: %s", _encoder_cache)
        return _encoder_cache


def _encode_args() -> list[str]:
    """Parâmetros de encode. IDÊNTICOS em todo segmento — é o que permite juntar
    os pedaços com `concat -c copy`, sem re-codificar a timeline inteira."""
    if video_encoder() == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
            "-cq", str(FINAL_CRF + 3), "-b:v", "0",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264", "-preset", FINAL_PRESET, "-crf", str(FINAL_CRF),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ]


# --- ffmpeg ----------------------------------------------------------------


def _run_ffmpeg(
    cmd: list[str],
    *,
    tag: str,
    duration: float,
    on_progress,
    cancelled,
) -> None:
    """Roda um ffmpeg reportando progresso e obedecendo a cancelamento.

    O `-progress pipe:1` cospe `out_time_ms=` conforme codifica; é daí que sai a
    porcentagem real (nada de barra fingida). O stderr vai pra um arquivo
    temporário e só é lido se der errado — é onde mora o diagnóstico.
    """
    full = [*cmd, "-progress", "pipe:1", "-nostats"]
    log.debug("%s | %s", tag, " ".join(full))
    err_f = tempfile.TemporaryFile()
    started = time.monotonic()
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=err_f, text=True)
    assert proc.stdout is not None
    last_check = 0.0
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms=") and duration > 0:
                try:
                    ms = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                on_progress(max(0.0, min(1.0, (ms / 1e6) / duration)))
            # O cancelamento é consultado no banco, então tem custo: no máximo
            # uma consulta a cada 5s, e só enquanto o ffmpeg fala.
            now = time.monotonic()
            if now - last_check > 5:
                last_check = now
                if cancelled():
                    proc.kill()
                    raise RenderCancelled()
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if code != 0:
        err_f.seek(0)
        tail = err_f.read().decode("utf-8", "replace")[-1500:]
        err_f.close()
        log.error("%s | ffmpeg saiu %s em %.1fs\n%s", tag, code, time.monotonic() - started, tail)
        raise RuntimeError(f"ffmpeg falhou ({code}) em {tag}.")
    err_f.close()
    log.info("%s | ok em %.1fs", tag, time.monotonic() - started)


def _atempo_chain(speed: float) -> list[str]:
    """`atempo` só aceita 0.5–2.0 por instância; fora disso encadeia.

    O editor permite 0.5–4×, então 4× vira `atempo=2,atempo=2`. Sem isso o
    ffmpeg recusa o filtro e o áudio do clip acelerado somem."""
    if abs(speed - 1.0) < 1e-6:
        return []
    out = []
    s = speed
    while s > 2.0:
        out.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        out.append("atempo=0.5")
        s *= 2.0
    out.append(f"atempo={s:.6f}")
    return out


def _layer_chain(lay: VideoLayer, label_in: str, label_out: str, fps: int, speed: float) -> str:
    """Filtro de uma fonte de vídeo: velocidade → recorte → escala → espelho →
    correção de imagem → RGBA. A ordem espelha a do compositor do canvas
    (`drawContain`/`drawCover` + `ctx.filter`), pra saída e preview baterem."""
    steps = [f"fps={fps}", f"setpts=(PTS-STARTPTS)/{speed:.6f}"]
    if lay.crop:
        c = lay.crop
        steps.append(
            f"crop=iw*{c.w:.6f}:ih*{c.h:.6f}:iw*{c.x:.6f}:ih*{c.y:.6f}"
        )
    w, h = max(2, round(lay.rect.w)), max(2, round(lay.rect.h))
    if lay.fit == "cover":
        # Preenche recortando o excedente pelo centro — o mesmo que o `drawCover`.
        steps.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
        steps.append(f"crop={w}:{h}")
    else:
        # `contain`: o rect já veio com a proporção certa do Class.
        steps.append(f"scale={w}:{h}")
    if lay.mirror:
        steps.append("hflip")
    # `brightness` do CSS é MULTIPLICATIVO; o `brightness` do filtro `eq` é
    # aditivo. Usar `eq` aqui deixaria a câmera diferente do preview — por isso o
    # brilho vai no colorchannelmixer e só o contraste (que no `eq` já é
    # multiplicativo em torno do cinza médio, igual ao CSS) vai no `eq`.
    if abs(lay.brightness - 1.0) > 1e-6:
        b = lay.brightness
        steps.append(f"colorchannelmixer=rr={b:.4f}:gg={b:.4f}:bb={b:.4f}")
    if abs(lay.contrast - 1.0) > 1e-6:
        steps.append(f"eq=contrast={lay.contrast:.4f}")
    steps.append("format=rgba")
    return f"[{label_in}]" + ",".join(steps) + f"[{label_out}]"


def _segment_cmd(
    plan: RenderPlan,
    seg: Segment,
    local: dict[str, str],
    out_path: str,
) -> list[str]:
    """Monta o ffmpeg de UM segmento: PNG de fundo + camadas de vídeo + PNG da
    frente + fade, encodado com os parâmetros comuns a todos os segmentos."""
    fps = plan.fps
    inputs: list[str] = []
    filters: list[str] = []
    idx = 0

    # 0: a base (fundo + sombras). É um PNG parado esticado pela duração do
    # segmento — é ele que define o comprimento da saída.
    inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{seg.dur:.6f}",
               "-i", local[seg.under_url]]
    filters.append(f"[0:v]format=rgba,fps={fps}[base]")
    idx += 1
    cur = "base"

    src_dur = seg.dur * seg.speed
    for n, lay in enumerate(seg.layers):
        # Corte na ENTRADA (`-ss` antes do `-i`): o ffmpeg pula direto pro
        # keyframe mais próximo em vez de decodificar tudo desde o começo. As
        # fontes preparadas têm keyframe a cada 0,5s, então o desperdício é
        # meio segundo por segmento.
        vi = idx
        inputs += ["-ss", f"{lay.start:.6f}", "-t", f"{src_dur:.6f}",
                   "-i", local[plan.sources[lay.source]]]
        idx += 1
        filters.append(_layer_chain(lay, f"{vi}:v", f"l{n}", fps, seg.speed))
        src_label = f"l{n}"
        if lay.mask_url:
            mi = idx
            inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{seg.dur:.6f}",
                       "-i", local[lay.mask_url]]
            idx += 1
            w, h = max(2, round(lay.rect.w)), max(2, round(lay.rect.h))
            filters.append(f"[{mi}:v]scale={w}:{h},format=gray[m{n}]")
            filters.append(f"[l{n}][m{n}]alphamerge[l{n}a]")
            src_label = f"l{n}a"
        x, y = round(lay.rect.x), round(lay.rect.y)
        filters.append(f"[{cur}][{src_label}]overlay={x}:{y}[o{n}]")
        cur = f"o{n}"

    if seg.over_url:
        oi = idx
        inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{seg.dur:.6f}",
                   "-i", local[seg.over_url]]
        idx += 1
        filters.append(f"[{oi}:v]format=rgba[ov]")
        filters.append(f"[{cur}][ov]overlay=0:0[ow]")
        cur = "ow"

    tail = []
    if seg.fade_in > 0.001:
        tail.append(f"fade=t=in:st=0:d={seg.fade_in:.4f}")
    if seg.fade_out > 0.001:
        st = max(0.0, seg.dur - seg.fade_out)
        tail.append(f"fade=t=out:st={st:.4f}:d={seg.fade_out:.4f}")
    tail.append("format=yuv420p")
    filters.append(f"[{cur}]" + ",".join(tail) + "[vout]")

    return [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-an",
        "-t", f"{seg.dur:.6f}",
        *_encode_args(),
        "-r", str(fps), "-vsync", "cfr",
        "-movflags", "+faststart",
        out_path,
    ]


def _audio_cmd(plan: RenderPlan, local: dict[str, str], out_path: str) -> list[str] | None:
    """Monta o ffmpeg do áudio da timeline inteira numa passada só.

    Cada trecho entra como uma ENTRADA própria (o mesmo arquivo pode aparecer
    várias vezes). Parece desperdício, mas é o contrário: com `asplit` o ffmpeg
    seguraria na RAM os frames dos ramos que ainda não foram consumidos, e áudio
    de aula longa em PCM é medido em centenas de MB.
    """
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    idx = 0

    for n, c in enumerate(plan.audio_clips):
        if c.gain <= 0 or c.dur <= 0:
            continue  # mudo não vira entrada: menos trabalho e nenhum ruído de borda
        src_dur = c.dur * c.speed
        inputs += ["-ss", f"{c.start:.6f}", "-t", f"{src_dur:.6f}",
                   "-i", local[plan.sources[c.source]]]
        steps = ["asetpts=PTS-STARTPTS", *_atempo_chain(c.speed)]
        if abs(c.gain - 1.0) > 1e-6:
            steps.append(f"volume={c.gain:.4f}")
        if c.at > 0.001:
            steps.append(f"adelay={round(c.at * 1000)}:all=1")
        steps.append(f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo")
        filters.append(f"[{idx}:a]" + ",".join(steps) + f"[ac{n}]")
        labels.append(f"ac{n}")
        idx += 1

    for n, b in enumerate(plan.audio_blocks):
        if b.gain <= 0 or b.dur <= 0:
            continue
        pre = ["-stream_loop", "-1"] if b.loop else []
        inputs += [*pre, "-ss", f"{b.offset:.6f}", "-t", f"{b.dur:.6f}",
                   "-i", local[plan.sources[b.source]]]
        steps = ["asetpts=PTS-STARTPTS"]
        if abs(b.gain - 1.0) > 1e-6:
            steps.append(f"volume={b.gain:.4f}")
        # Fades ANTES do adelay: os tempos do envelope são relativos ao começo do
        # bloco, não ao começo da timeline.
        if b.fade_in > 0.001:
            steps.append(f"afade=t=in:st=0:d={b.fade_in:.4f}")
        if b.fade_out > 0.001:
            steps.append(f"afade=t=out:st={max(0.0, b.dur - b.fade_out):.4f}:d={b.fade_out:.4f}")
        if b.at > 0.001:
            steps.append(f"adelay={round(b.at * 1000)}:all=1")
        steps.append(f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo")
        filters.append(f"[{idx}:a]" + ",".join(steps) + f"[ab{n}]")
        labels.append(f"ab{n}")
        idx += 1

    if not labels:
        return None

    mix = "".join(f"[{l}]" for l in labels)
    # `normalize=0`: o amix, por padrão, DIVIDE pelo número de entradas — a voz
    # perderia metade do volume só por existir uma música de fundo. Aqui as
    # faixas somam, como no Web Audio do preview.
    filters.append(f"{mix}amix=inputs={len(labels)}:normalize=0:duration=longest[sum]")
    last = "sum"
    if plan.limiter:
        # Somar pode estourar o teto mesmo com todos os ganhos abaixo de 100%.
        # O limitador é o que faz "mais alto" não virar distorção.
        filters.append(f"[{last}]alimiter=limit=0.891:attack=3:release=250[lim]")
        last = "lim"
    # `apad` + `-t`: se o áudio acabar antes do vídeo, o arquivo final fica com
    # faixas de durações diferentes e alguns players mostram a duração errada.
    filters.append(f"[{last}]apad,atrim=0:{plan.duration:.6f},asetpts=PTS-STARTPTS[aout]")

    return [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-t", f"{plan.duration:.6f}",
        out_path,
    ]


# --- Job -------------------------------------------------------------------


def _download(url: str, dest: str) -> int:
    """Baixa em streaming pro disco. Nunca na memória: as fontes de uma aula são
    centenas de MB e o serviço atende outros jobs ao mesmo tempo."""
    total = 0
    with requests.get(url, stream=True, timeout=(30, 900)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    return total


def _run_plan(plan: RenderPlan) -> None:
    tag = f"render {plan.recording_id}"
    started = time.monotonic()
    work = tempfile.mkdtemp(prefix="render-")
    progress = {"value": 0.0}

    def report(value: float) -> None:
        # Só grava quando muda pelo menos 1 ponto percentual: o ffmpeg cospe
        # progresso várias vezes por segundo e cada gravação é um round-trip no
        # MySQL remoto.
        v = max(0.0, min(0.999, value))
        if v - progress["value"] < 0.01:
            return
        progress["value"] = v
        db.update_export(plan.recording_id, progress=v)

    def cancelled() -> bool:
        return db.export_cancelled(plan.recording_id)

    def guard() -> None:
        if cancelled():
            raise RenderCancelled()

    try:
        db.update_export(plan.recording_id, status="RENDERING", progress=0.0)
        log.info(
            "%s | início (%.1fs, %d segmento(s), %d faixa(s) de áudio)",
            tag, plan.duration, len(plan.segments), len(plan.audio_clips) + len(plan.audio_blocks),
        )

        # 1) Fontes e camadas, uma vez cada. Ler direto das URLs em cada
        # segmento faria o MinIO servir o mesmo arquivo N vezes e ainda deixaria
        # o render refém do prazo da URL assinada.
        every = list(dict.fromkeys([*plan.sources.values(), *(
            u for s in plan.segments
            for u in ([s.under_url, s.over_url] + [l.mask_url for l in s.layers])
            if u
        )]))
        local: dict[str, str] = {}
        for n, url in enumerate(every):
            guard()
            # A extensão sai do caminho da URL. Não é enfeite: o ffmpeg escolhe o
            # demuxer de imagem pelo nome quando o arquivo é entrada de `-loop 1`.
            ext = os.path.splitext(urlparse(url).path)[1][:6] or ".bin"
            dest = os.path.join(work, f"src{n}{ext}")
            _download(url, dest)
            local[url] = dest
            report(W_DOWNLOAD * (n + 1) / max(1, len(every)))
        base = W_DOWNLOAD

        # 2) Um MP4 por segmento, todos com o mesmo encode.
        seg_paths: list[str] = []
        done = 0.0
        for n, seg in enumerate(plan.segments):
            guard()
            out = os.path.join(work, f"seg{n:04d}.mp4")
            share = (seg.dur / plan.duration) if plan.duration > 0 else 1.0
            _run_ffmpeg(
                _segment_cmd(plan, seg, local, out),
                tag=f"{tag} [segmento {n + 1}/{len(plan.segments)}]",
                duration=seg.dur,
                on_progress=lambda f, d=done, s=share: report(base + W_VIDEO * (d + f * s)),
                cancelled=cancelled,
            )
            done += share
            seg_paths.append(out)
            report(base + W_VIDEO * done)
        base += W_VIDEO

        # 3) Junta sem re-codificar (os parâmetros são idênticos por construção).
        guard()
        if len(seg_paths) == 1:
            video_path = seg_paths[0]
        else:
            video_path = os.path.join(work, "video.mp4")
            list_path = os.path.join(work, "segments.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in seg_paths:
                    f.write(f"file '{p}'\n")
            _run_ffmpeg(
                ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c", "copy", "-movflags", "+faststart", video_path],
                tag=f"{tag} [juntando]",
                duration=plan.duration,
                on_progress=lambda f: None,
                cancelled=cancelled,
            )

        # 4) Áudio da timeline inteira numa passada.
        guard()
        audio_cmd = _audio_cmd(plan, local, os.path.join(work, "audio.m4a"))
        audio_path = None
        if audio_cmd:
            audio_path = os.path.join(work, "audio.m4a")
            _run_ffmpeg(
                audio_cmd,
                tag=f"{tag} [áudio]",
                duration=plan.duration,
                on_progress=lambda f: report(base + W_AUDIO * f),
                cancelled=cancelled,
            )
        report(base + W_AUDIO)
        base += W_AUDIO

        # 5) Mux final: `-c copy` nos dois lados, então é I/O, não encode.
        guard()
        final = os.path.join(work, "export.mp4")
        if audio_path:
            _run_ffmpeg(
                ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", video_path, "-i", audio_path,
                 "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                 "-movflags", "+faststart", final],
                tag=f"{tag} [finalizando]",
                duration=plan.duration,
                on_progress=lambda f: report(base + W_MUX * f),
                cancelled=cancelled,
            )
        else:
            shutil.move(video_path, final)
        # O que sobra (W_UPLOAD) é o envio pro MinIO, que não tem progresso
        # granular: a barra fica parada nos últimos por cento até o PUT voltar.
        report(1.0 - W_UPLOAD)

        # 6) Sobe pro MinIO. O arquivo EXISTIR é a verdade do "pronto" — o estado
        # no banco é conveniência de interface, e o Class reconcilia por ele.
        guard()
        size = os.path.getsize(final)
        with open(final, "rb") as f:
            put = requests.put(
                plan.upload_url,
                data=f,
                headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
                timeout=(30, 1800),
            )
        if put.status_code not in (200, 204):
            raise RuntimeError(f"PUT no MinIO falhou ({put.status_code}).")

        db.update_export(
            plan.recording_id,
            status="READY",
            progress=1.0,
            spec_hash=plan.spec_hash,
            exported_at=datetime.now(timezone.utc),
        )
        log.info(
            "%s | pronto em %.1fs (%.1f MB, %.1f× tempo real)",
            tag, time.monotonic() - started, size / (1024 * 1024),
            plan.duration / max(0.001, time.monotonic() - started),
        )
    except RenderCancelled:
        # NÃO mexe no estado: quem cancelou já escreveu o dele, e sobrescrever
        # aqui apagaria o QUEUED de um reexport que acabou de entrar.
        log.info("%s | cancelado", tag)
    except Exception as e:  # noqa: BLE001
        log.exception("%s | FALHOU em %.1fs", tag, time.monotonic() - started)
        db.update_export(
            plan.recording_id, status="FAILED", progress=0.0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Fila ------------------------------------------------------------------

_queue: "queue.Queue[RenderPlan]" = queue.Queue(maxsize=RENDER_QUEUE_MAX)
_active: set[str] = set()
_active_lock = threading.Lock()
_workers_started = False


def _worker(n: int) -> None:
    while True:
        plan = _queue.get()
        try:
            _run_plan(plan)
        except Exception:  # noqa: BLE001
            log.exception("render | worker %d morreu processando %s", n, plan.recording_id)
        finally:
            with _active_lock:
                _active.discard(plan.recording_id)
            _queue.task_done()


def start_workers() -> None:
    """Sobe os workers uma vez, no import. São daemon: não seguram o shutdown."""
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    for n in range(max(1, RENDER_CONCURRENCY)):
        threading.Thread(target=_worker, args=(n,), daemon=True, name=f"render-{n}").start()
    log.info(
        "render | %d worker(s), fila máx %d, encoder detectado sob demanda",
        RENDER_CONCURRENCY, RENDER_QUEUE_MAX,
    )


def enqueue(plan: RenderPlan) -> str:
    """Coloca o plano na fila. Devolve "queued", "duplicate" ou "full"."""
    with _active_lock:
        if plan.recording_id in _active:
            return "duplicate"
        try:
            _queue.put_nowait(plan)
        except queue.Full:
            return "full"
        _active.add(plan.recording_id)
    return "queued"


def pending() -> int:
    return _queue.qsize()
