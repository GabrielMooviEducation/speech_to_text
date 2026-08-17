import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()  # carrega variáveis do arquivo .env, se existir

# --- Logs -----------------------------------------------------------------
# Tudo em stdout pra sair no `docker logs` do serviço. Quando uma transcrição
# falha, o moovi_class só recebe a mensagem curta do callback — o diagnóstico
# completo (etapa, tempo, stderr do ffmpeg, corpo do erro da API) fica aqui.
# LOG_LEVEL=DEBUG liga os tracebacks de erros já esperados (HTTPException).
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # não deixa a config do uvicorn engolir a nossa
)
log = logging.getLogger("speech")

MB = 1024 * 1024


def describe_error(e: BaseException) -> str:
    """Mensagem curta e informativa de uma exceção.

    `str(e)` de uma HTTPException vira "500: ffmpeg falhou..." (útil), mas de um
    KeyError vira só "'GOOGLE_SA_JSON'" — sem o tipo, ninguém entende. Por isso
    prefixamos com o nome da classe nos erros que não são HTTP.
    """
    if isinstance(e, StarletteHTTPException):
        return f"HTTP {e.status_code}: {e.detail}"
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


@contextmanager
def timed(step: str, tag: str, ctx: dict | None = None):
    """Cronometra uma etapa e loga o fim (ok ou falha) sem mudar a exceção.

    `ctx["step"]` guarda a etapa atual pra que, lá no fim, o erro enviado ao
    moovi_class diga ONDE quebrou (download do Drive? ffmpeg? OpenAI?).
    """
    if ctx is not None:
        ctx["step"] = step
    started = time.monotonic()
    log.info("%s | %s | início", tag, step)
    try:
        yield
    except StarletteHTTPException as e:
        log.error(
            "%s | %s | FALHOU em %.1fs: %s",
            tag, step, time.monotonic() - started, describe_error(e),
        )
        log.debug("%s | %s | traceback", tag, step, exc_info=True)
        raise
    except Exception:  # noqa: BLE001 — erro inesperado: traceback completo
        log.exception(
            "%s | %s | FALHOU em %.1fs (erro inesperado)",
            tag, step, time.monotonic() - started,
        )
        raise
    log.info("%s | %s | ok em %.1fs", tag, step, time.monotonic() - started)


app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Uma linha por request (método, rota, status, tempo) + qualquer exceção
    que escape dos handlers — é o que falta hoje pra saber que a chamada
    sequer chegou no serviço."""
    rid = uuid.uuid4().hex[:6]
    started = time.monotonic()
    log.info("req %s | %s %s", rid, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        log.exception(
            "req %s | %s %s | EXCEÇÃO NÃO TRATADA em %.1fs",
            rid, request.method, request.url.path, time.monotonic() - started,
        )
        raise
    log.info(
        "req %s | %s %s | %s em %.1fs",
        rid, request.method, request.url.path,
        response.status_code, time.monotonic() - started,
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def log_http_exception(request: Request, exc: StarletteHTTPException):
    """Todo 4xx/5xx sai no console com o motivo — hoje o detalhe some na
    resposta HTTP e nada fica registrado no servidor."""
    log.warning(
        "%s %s | %s: %s", request.method, request.url.path, exc.status_code, exc.detail
    )
    return await http_exception_handler(request, exc)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# --- Escolha do motor de transcrição --------------------------------------
# Por padrão usa o faster-whisper LOCAL. Setando USE_OPENAI=true no .env, usa a
# API de transcrição da OpenAI (precisa de OPENAI_API_KEY). Assim dá pra rodar
# sem baixar o modelo grande / sem CPU pesada.
USE_OPENAI = os.getenv("USE_OPENAI", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Aceita OPENAI_KEY (nome usado no .env) ou OPENAI_API_KEY como fallback.
OPENAI_API_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "whisper-1")

_local_model = None


def get_local_model():
    """Carrega o WhisperModel local só na 1ª vez (lazy) pra não segurar RAM
    quando o serviço está configurado pra usar a OpenAI."""
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel

        # 1ª transcrição depois do deploy baixa ~3GB de modelo: sem este log
        # parece que o job travou.
        log.info("carregando WhisperModel large-v3 (1ª vez pode baixar o modelo)...")
        started = time.monotonic()
        _local_model = WhisperModel(
            "large-v3", device="cpu", compute_type="int8"
        )
        log.info("WhisperModel carregado em %.1fs", time.monotonic() - started)
    return _local_model


# Segredo compartilhado com o moovi_class: vai no header do callback pra que o
# receptor confirme que a chamada veio mesmo deste serviço.
CALLBACK_SECRET = os.getenv("TRANSCRIPTION_CALLBACK_SECRET")

# URL FIXA do callback no moovi_class. Definida por env do PRÓPRIO serviço — NÃO
# vem na request — pra não virar SSRF nem oráculo de exfiltração do secret (um
# chamador não pode redirecionar o callback/secret pra um host arbitrário).
CALLBACK_URL = os.getenv("TRANSCRIPTION_CALLBACK_URL")


def load_sa_info():
    raw = os.getenv("GOOGLE_SA_JSON", "").strip()
    if not raw:
        log.error(
            "GOOGLE_SA_JSON ausente — sem credencial não dá pra baixar do Drive."
        )
        raise RuntimeError("GOOGLE_SA_JSON não configurada.")
    try:
        return json.loads(raw)  # JSON cru
    except json.JSONDecodeError:
        try:
            return json.loads(base64.b64decode(raw))  # base64 do JSON
        except Exception as e:  # noqa: BLE001
            log.error("GOOGLE_SA_JSON inválida (nem JSON, nem base64 de JSON).")
            raise RuntimeError(f"GOOGLE_SA_JSON inválida: {describe_error(e)}") from e


creds = service_account.Credentials.from_service_account_info(
    load_sa_info(), scopes=SCOPES
)


DEFAULT_PROMPT = os.getenv(
    "DEFAULT_PROMPT",
    "Transcrição de um vídeo em português do Brasil.",
)


def _ffmpeg_version() -> str:
    """Confere na subida se o ffmpeg existe — sem ele TODA transcrição falha, e
    o erro só apareceria no meio de um job em background."""
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        first = proc.stdout.decode("utf-8", "ignore").splitlines()
        return first[0] if first else "presente"
    except Exception as e:  # noqa: BLE001
        return f"AUSENTE ({describe_error(e)})"


def _log_config() -> None:
    log.info("=" * 60)
    log.info("speech_to_text no ar | log_level=%s", LOG_LEVEL)
    log.info(
        "motor=%s | modelo=%s",
        "openai" if USE_OPENAI else "local (faster-whisper)",
        OPENAI_MODEL if USE_OPENAI else "large-v3",
    )
    if USE_OPENAI and not OPENAI_API_KEY:
        log.error("USE_OPENAI=true mas OPENAI_KEY/OPENAI_API_KEY ausente.")
    log.info("callback=%s", CALLBACK_URL or "NÃO CONFIGURADA (/transcribe-async dá 503)")
    if not CALLBACK_SECRET:
        log.warning(
            "TRANSCRIPTION_CALLBACK_SECRET ausente — o moovi_class responde 401 "
            "no callback e a aula fica presa em PROCESSING."
        )
    ffmpeg = _ffmpeg_version()
    if ffmpeg.startswith("AUSENTE"):
        log.error("ffmpeg=%s — nenhuma transcrição/remux vai funcionar.", ffmpeg)
    else:
        log.info("ffmpeg=%s", ffmpeg)
    log.info("=" * 60)


_log_config()


class Body(BaseModel):
    file_id: str
    prompt: str | None = None


class AsyncBody(BaseModel):
    file_id: str
    lesson_id: str
    prompt: str | None = None


def get_token() -> str:
    if not creds.valid:
        creds.refresh(GoogleRequest())
    return creds.token


def download_from_drive(file_id: str, dest: str) -> int:
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    params = {"alt": "media", "supportsAllDrives": "true"}
    with requests.get(url, headers=headers, params=params, stream=True) as r:
        if r.status_code != 200:
            # No log vai o corpo inteiro (permissão da service account, quota,
            # arquivo removido...); na resposta HTTP continua truncado.
            log.error("Drive recusou %s (file_id=%s): %s", r.status_code, file_id, r.text)
            raise HTTPException(
                status_code=400,
                detail=f"Drive recusou ({r.status_code}): {r.text[:500]}",
            )
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                total += len(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Drive devolveu arquivo vazio.")
    return total


# Limite duro da API de transcrição da OpenAI.
OPENAI_MAX_BYTES = 25 * 1024 * 1024


def extract_audio(video_path: str) -> str:
    """Extrai o áudio do vídeo em mono/16kHz/MP3 comprimido via ffmpeg.

    Whisper (local e OpenAI) só usa áudio, e 16kHz mono é o formato ideal pro
    modelo. Comprimir aqui derruba o tamanho de centenas de MB (vídeo) pra ~14MB
    por hora de aula — o que mantém a chamada da OpenAI abaixo do teto de 25MB.
    """
    audio_path = video_path + ".mp3"
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
            audio_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        # O erro real do ffmpeg sai no FIM do stderr; o começo é só o banner de
        # versão/config. Por isso pegamos a cauda (-500), não a cabeça.
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        log.error("ffmpeg (extract_audio) saiu %s. stderr:\n%s", proc.returncode, stderr[-4000:])
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg falhou: {stderr[-500:]}",
        )
    return audio_path


def transcribe_local(path: str, prompt: str | None) -> str:
    segments, _ = get_local_model().transcribe(
        path,
        language="pt",
        vad_filter=True,
        initial_prompt=prompt or DEFAULT_PROMPT,
    )
    return " ".join(seg.text.strip() for seg in segments)


def transcribe_openai(path: str, prompt: str | None) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="USE_OPENAI=true mas OPENAI_KEY não configurada.",
        )
    # Recebe o áudio já extraído/comprimido; ainda assim protege o teto de 25MB
    # (aulas muito longas podem estourar mesmo comprimidas).
    size = os.path.getsize(path)
    if size > OPENAI_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Áudio de {size // (1024 * 1024)}MB excede o limite de 25MB da "
                "OpenAI. Use o motor local (USE_OPENAI=false) para esta aula."
            ),
        )
    with open(path, "rb") as f:
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={
                "model": OPENAI_MODEL,
                "language": "pt",
                "prompt": prompt or DEFAULT_PROMPT,
            },
            files={"file": (os.path.basename(path), f, "audio/mpeg")},
            timeout=600,
        )
    if resp.status_code != 200:
        log.error("OpenAI recusou %s. Corpo:\n%s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI recusou ({resp.status_code}): {resp.text[:500]}",
        )
    try:
        return resp.json()["text"].strip()
    except (ValueError, KeyError) as e:
        log.error("Resposta inesperada da OpenAI: %s", resp.text[:2000])
        raise HTTPException(
            status_code=502,
            detail=f"Resposta inesperada da OpenAI ({describe_error(e)}).",
        ) from e


def run_transcription(
    file_id: str, prompt: str | None, tag: str = "sync", ctx: dict | None = None
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    audio_path = None
    engine = f"openai/{OPENAI_MODEL}" if USE_OPENAI else "local/large-v3"
    try:
        with timed("download do Drive", tag, ctx):
            size = download_from_drive(file_id, video_path)
            log.info("%s | vídeo baixado: %.1f MB", tag, size / MB)

        with timed("extração de áudio (ffmpeg)", tag, ctx):
            audio_path = extract_audio(video_path)
            log.info("%s | áudio: %.1f MB", tag, os.path.getsize(audio_path) / MB)

        with timed(f"transcrição ({engine})", tag, ctx):
            text = (
                transcribe_openai(audio_path, prompt)
                if USE_OPENAI
                else transcribe_local(audio_path, prompt)
            )
        log.info("%s | texto com %d caracteres", tag, len(text))
        if not text.strip():
            # O moovi_class trata texto vazio como falha; sem este aviso o log
            # mostraria "ok" e a aula apareceria FAILED sem explicação.
            log.warning("%s | transcrição VAZIA (áudio mudo?)", tag)
        return text
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(video_path):
            os.remove(video_path)


@app.post("/transcribe")
def transcribe(body: Body):
    return {"text": run_transcription(body.file_id, body.prompt)}


def process_and_callback(file_id: str, prompt: str | None, lesson_id: str):
    """Transcreve em background e devolve o resultado pro moovi_class via POST.

    Roda DEPOIS da resposta 202 — a transcrição pode levar minutos sem prender
    nenhuma requisição. Sucesso envia {lesson_id, text}; falha envia
    {lesson_id, error}. O moovi_class persiste o texto e atualiza o status.
    O destino é a CALLBACK_URL fixa do serviço (não vem da request).
    """
    tag = f"lesson={lesson_id}"
    ctx = {"step": "início"}
    started = time.monotonic()
    log.info("%s | JOB INICIADO (file_id=%s)", tag, file_id)

    try:
        text = run_transcription(file_id, prompt, tag, ctx)
        payload = {"lesson_id": lesson_id, "text": text}
        log.info(
            "%s | JOB OK em %.1fs (%d caracteres)",
            tag, time.monotonic() - started, len(text),
        )
    except Exception as e:  # noqa: BLE001 — qualquer falha vira status FAILED
        # A etapa vai junto na mensagem: é ela que aparece no log do moovi_class
        # ("[extração de áudio (ffmpeg)] HTTP 500: ...") e diz onde procurar aqui.
        message = f"[{ctx['step']}] {describe_error(e)}"
        log.error("%s | JOB FALHOU em %.1fs: %s", tag, time.monotonic() - started, message)
        payload = {"lesson_id": lesson_id, "error": message[:500]}

    headers = {"Content-Type": "application/json"}
    if CALLBACK_SECRET:
        headers["X-Callback-Secret"] = CALLBACK_SECRET
    try:
        resp = requests.post(CALLBACK_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code >= 300:
            # 401 aqui = secret divergente entre os dois serviços; a aula fica
            # em PROCESSING pra sempre e nada aparece na tela do professor.
            log.error(
                "%s | callback recusado (%s): %s", tag, resp.status_code, resp.text[:500]
            )
        else:
            log.info("%s | callback entregue (%s)", tag, resp.status_code)
    except requests.RequestException as e:
        # callback inalcançável: o moovi_class fica em PROCESSING (reprocessável)
        log.error(
            "%s | callback inalcançável (%s): %s — a aula fica em PROCESSING",
            tag, CALLBACK_URL, describe_error(e),
        )


# --- Remux (conserta a barra de progresso dos vídeos exportados) ----------
# O MediaRecorder do navegador grava "ao vivo" e NÃO escreve a duração no
# cabeçalho do arquivo — por isso o player não mostra progresso (cursor preso no
# início). Aqui reescrevemos o container SEM re-encodar (`-c copy`, rápido) e, no
# mp4, com `+faststart` (move o índice pro começo) — o arquivo fica seekável.
# Reaproveita o ffmpeg que já existe pra transcrição.


@app.post("/remux")
async def remux(request: Request, ext: str = "mp4"):
    ext = "webm" if ext.lower() == "webm" else "mp4"
    suffix = f".{ext}"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tin:
        in_path = tin.name
        wrote = 0
        async for chunk in request.stream():
            tin.write(chunk)
            wrote += len(chunk)
    if wrote == 0:
        os.remove(in_path)
        raise HTTPException(status_code=400, detail="Corpo vazio.")

    out_path = in_path + ".out" + suffix
    cmd = ["ffmpeg", "-y", "-i", in_path, "-c", "copy"]
    if ext == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [out_path]

    proc = subprocess.run(cmd, capture_output=True)
    os.remove(in_path)
    if proc.returncode != 0 or not os.path.exists(out_path):
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        log.error("ffmpeg (remux %s) saiu %s. stderr:\n%s", ext, proc.returncode, stderr[-4000:])
        if os.path.exists(out_path):
            os.remove(out_path)
        raise HTTPException(status_code=500, detail=f"ffmpeg falhou: {stderr[-500:]}")

    media = "video/mp4" if ext == "mp4" else "video/webm"
    # Remove o arquivo de saída DEPOIS que a resposta terminar de ser enviada.
    return FileResponse(
        out_path,
        media_type=media,
        filename=f"remux.{ext}",
        background=BackgroundTask(os.remove, out_path),
    )


# --- Onda do áudio (picos) ------------------------------------------------
# O editor mostra a forma de onda do áudio. Gerar isso no navegador exigiria
# baixar o vídeo inteiro (centenas de MB) e decodificar — trava. Aqui o ffmpeg lê
# a URL do MinIO, extrai o áudio (mono/8kHz) e devolvemos só os PICOS normalizados
# (um JSON pequeno). O moovi_class chama uma vez e cacheia.


class PeaksBody(BaseModel):
    url: str
    buckets: int | None = None


def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    # Bloqueia hosts internos (SSRF): localhost, loopback, redes privadas.
    if host in ("localhost", "0.0.0.0"):
        return False
    if re.match(r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)", host):
        return False
    if host.endswith(".internal") or host.endswith(".local"):
        return False
    return True


@app.post("/peaks")
def peaks(body: PeaksBody):
    if not _is_safe_url(body.url):
        raise HTTPException(status_code=400, detail="URL não permitida.")
    buckets = max(50, min(body.buckets or 900, 4000))

    # ffmpeg lê a URL direto e joga PCM (s16le mono 8kHz) no stdout.
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-i", body.url,
            "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        log.error("ffmpeg (peaks) saiu %s. stderr:\n%s", proc.returncode, stderr[-4000:])
        raise HTTPException(status_code=500, detail=f"ffmpeg falhou: {stderr[-400:]}")

    import numpy as np

    raw = proc.stdout
    arr = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype=np.int16)
    if arr.size == 0:
        return {"peaks": []}
    arr = np.abs(arr.astype(np.float32))
    n_buckets = min(buckets, arr.size)
    per = arr.size // n_buckets
    trimmed = arr[: per * n_buckets].reshape(n_buckets, per)
    peaks = trimmed.max(axis=1)
    top = float(peaks.max()) or 1.0
    return {"peaks": [round(float(p) / top, 4) for p in peaks]}


# --- Remux por URL (conserta as fontes: duração + índice/cues) ------------
# O WebM do MediaRecorder vem com duration=Infinity e sem cues → trava seek e
# playback. Aqui o ffmpeg lê a fonte no MinIO, reescreve o container (-c copy,
# sem re-encode) num arquivo temporário SEEKÁVEL (aí escreve os cues) e faz PUT
# direto no MinIO (URL presigned). O arquivo grande NÃO passa pelo moovi_class.


class RemuxUrlBody(BaseModel):
    url: str
    upload_url: str
    ext: str | None = None


@app.post("/remux-url")
def remux_url(body: RemuxUrlBody):
    if not _is_safe_url(body.url) or not _is_safe_url(body.upload_url):
        raise HTTPException(status_code=400, detail="URL não permitida.")
    ext = "mp4" if (body.ext or "").lower() == "mp4" else "webm"

    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    out_path = tmp.name
    tmp.close()

    # `-fflags +genpts`: regenera timestamps (a tela do MediaRecorder é VFR e às
    # vezes o -c copy puro não fecha a duração). Se o copy falhar, o chamador vê o
    # stderr e a gente decide re-encodar.
    cmd = ["ffmpeg", "-nostdin", "-fflags", "+genpts", "-y", "-i", body.url, "-c", "copy"]
    if ext == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [out_path]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        if os.path.exists(out_path):
            os.remove(out_path)
        stderr = proc.stderr.decode("utf-8", "ignore").strip()
        log.error(
            "ffmpeg (remux-url %s) saiu %s. stderr:\n%s", ext, proc.returncode, stderr[-4000:]
        )
        raise HTTPException(status_code=500, detail=f"ffmpeg falhou: {stderr[-400:]}")

    try:
        content_type = "video/mp4" if ext == "mp4" else "video/webm"
        with open(out_path, "rb") as f:  # streaming (não carrega tudo na RAM)
            put = requests.put(
                body.upload_url,
                data=f,
                headers={"Content-Type": content_type},
                timeout=1800,
            )
        if put.status_code not in (200, 204):
            raise HTTPException(
                status_code=502,
                detail=f"PUT no MinIO falhou ({put.status_code}).",
            )
    finally:
        os.remove(out_path)

    return {"ok": True}


# --- Prepara as fontes (MP4/H.264, keyframes densos) -----------------------
# O WebM VP9 do MediaRecorder é ruim pra processar no cliente (Infinity, GOP
# esparso, sem demuxer maduro). Aqui re-encodamos pra MP4/H.264 com keyframe a
# cada 15 frames (-g 15) → seek rápido no cliente E pronto pro mp4box+VideoDecoder.
# PUT direto no MinIO (URL presigned). Arquivo grande NÃO passa pelo moovi_class.
#
# São DUAS saídas por fonte, tiradas do mesmo cru e com a MESMA timebase:
#   full-res (CRF 20) → é o que o EXPORT lê; manda na qualidade final.
#   proxy    (CRF 28, altura reduzida) → é o que o EDITOR toca; manda na
#                                        velocidade de seek.
# A timebase idêntica é o que garante que um corte marcado no proxy caia no mesmo
# frame no full-res — por isso `force_fps` vale igual pros dois. Mexer no fps de
# um sem mexer no outro dessincroniza os cortes.


PROXY_CRF = 28
FULL_CRF = 20
# Fatia da barra de progresso reservada ao proxy. Ele é bem mais barato que o
# full-res (menos pixel por frame), então fica com um pedaço pequeno.
PROXY_PROGRESS_SHARE = 0.25


class PrepareSourceBody(BaseModel):
    url: str
    # Saídas. Ambas opcionais, mas ao menos uma é obrigatória: uma gravação
    # antiga já tem o full-res gerado e só precisa do proxy — re-encodar o
    # full-res de novo nesse caso seria puro desperdício.
    upload_url: str | None = None
    proxy_upload_url: str | None = None
    duration_sec: float | None = None
    # Quando setado, reamostra o vídeo pra esse fps constante (CFR). O
    # MediaRecorder gera timestamps irregulares e o ffmpeg dropa frames (a tela
    # vira ~2fps e trava nas animações).
    force_fps: int | None = None
    # Altura alvo do proxy; a largura sai da proporção original, então o
    # compositor do Class enquadra exatamente igual (a conta dele é por fração
    # do canvas e por proporção da fonte, nunca por pixel).
    proxy_height: int | None = None
    # A tela é MUDA no Class (preview e export tiram o som da câmera), então o
    # proxy dela dispensa faixa de áudio.
    proxy_audio: bool = True


# Estado dos jobs de re-encode, em memória (chave = URL da fonte). Assíncrono pra
# o professor poder sair da tela — o job segue no servidor e o resultado vai pro
# MinIO (a existência do objeto é a verdade do "pronto"). `progress` 0..1.
_prepare_jobs: dict[str, dict] = {}


def _encode_and_put(
    src_url: str,
    upload_url: str,
    *,
    job_key: str,
    tag: str,
    duration_sec: float,
    force_fps: int | None,
    crf: int,
    scale_height: int | None = None,
    audio_bitrate: str | None = "128k",
    progress_from: float = 0.0,
    progress_to: float = 1.0,
) -> None:
    """Re-encoda `src_url` num MP4/H.264 seekável e faz PUT no `upload_url`.

    Levanta exceção em QUALQUER falha (ffmpeg ou PUT) — quem chama é que sabe se
    aquele passo era essencial. O `-g 15` (keyframe a cada meio segundo) é o que
    torna o seek rápido, tanto no <video> do editor quanto no VideoDecoder do
    export. O progresso é reportado dentro da faixa [progress_from, progress_to]
    pra que dois passos seguidos preencham uma barra só.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()
    err_f = tempfile.TemporaryFile()
    started = time.monotonic()

    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", src_url,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
    ]
    if scale_height:
        # `-2` deriva a largura da proporção mantendo-a par (exigência do
        # yuv420p). O `min(ih,...)` evita AUMENTAR uma fonte que já seja menor
        # que o alvo — um "proxy" maior que o original seria o oposto do ponto.
        # A vírgula vai escapada: dentro de um filtro ela separaria argumentos.
        cmd += ["-vf", f"scale=-2:min(ih\\,{scale_height})"]
    if audio_bitrate:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
    else:
        cmd += ["-an"]
    if force_fps:
        # CFR: reamostra num relógio constante. O ffmpeg dropa frames da tela por
        # causa dos timestamps irregulares do MediaRecorder; forçar fps constante
        # recupera o movimento das animações. Preserva a duração (não dessincroniza).
        cmd += ["-r", str(force_fps), "-vsync", "cfr"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_path]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_f, text=True)
        assert proc.stdout is not None
        span = max(0.0, progress_to - progress_from)
        for line in proc.stdout:  # -progress: linhas key=value
            line = line.strip()
            if not line.startswith("out_time_ms=") or duration_sec <= 0:
                continue
            try:
                ms = int(line.split("=", 1)[1])
            except ValueError:
                continue
            done = min(1.0, max(0.0, (ms / 1_000_000.0) / duration_sec))
            job = _prepare_jobs.get(job_key)
            # O job some se alguém limpar o estado no meio; sem o `if`, uma
            # gravação cancelada derrubaria o passo com KeyError.
            if job is not None:
                job["progress"] = min(0.99, progress_from + done * span)
        proc.wait()
        if proc.returncode != 0:
            err_f.seek(0)
            full = err_f.read().decode("utf-8", "ignore")
            log.error(
                "%s | ffmpeg saiu %s. stderr:\n%s", tag, proc.returncode, full[-4000:]
            )
            raise RuntimeError(f"ffmpeg: {full[-400:]}")

        with open(out_path, "rb") as f:  # streaming (não carrega tudo na RAM)
            put = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "video/mp4"},
                timeout=3600,
            )
        if put.status_code not in (200, 204):
            log.error(
                "%s | PUT no MinIO falhou (%s): %s", tag, put.status_code, put.text[:500]
            )
            raise RuntimeError(f"PUT {put.status_code}")

        log.info("%s | ok em %.1fs", tag, time.monotonic() - started)
    finally:
        err_f.close()
        if os.path.exists(out_path):
            os.remove(out_path)


def _run_prepare(
    url: str,
    upload_url: str | None,
    duration_sec: float,
    force_fps: int | None = None,
    proxy_upload_url: str | None = None,
    proxy_height: int | None = None,
    proxy_audio: bool = True,
):
    tag = f"prepare {os.path.basename(urlparse(url).path) or url[-40:]}"
    started = time.monotonic()
    want_proxy = bool(proxy_upload_url and proxy_height)
    log.info(
        "%s | início (duração=%.1fs, fps=%s, proxy=%s, full=%s)",
        tag,
        duration_sec,
        force_fps or "original",
        f"{proxy_height}p" if want_proxy else "não",
        "sim" if upload_url else "não",
    )

    # O proxy vem PRIMEIRO: é barato e destrava o editor enquanto o full-res
    # ainda está rodando (quem decide "pronto" é a existência do objeto no
    # MinIO, então o Class enxerga assim que ele aparece). Falha aqui é
    # degradação, não erro: o editor cai na fonte cheia e só perde velocidade.
    if want_proxy:
        try:
            _encode_and_put(
                url,
                proxy_upload_url,  # type: ignore[arg-type]  # want_proxy garante
                job_key=url,
                tag=f"{tag} [proxy]",
                duration_sec=duration_sec,
                force_fps=force_fps,
                crf=PROXY_CRF,
                scale_height=proxy_height,
                audio_bitrate="96k" if proxy_audio else None,
                progress_from=0.0,
                progress_to=PROXY_PROGRESS_SHARE,
            )
        except Exception:  # noqa: BLE001
            log.exception("%s [proxy] | falhou; seguindo só com o full-res", tag)

    if upload_url:
        try:
            _encode_and_put(
                url,
                upload_url,
                job_key=url,
                tag=f"{tag} [full]",
                duration_sec=duration_sec,
                force_fps=force_fps,
                crf=FULL_CRF,
                audio_bitrate="128k",
                progress_from=PROXY_PROGRESS_SHARE if want_proxy else 0.0,
                progress_to=1.0,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("%s | FALHOU em %.1fs", tag, time.monotonic() - started)
            _prepare_jobs[url] = {
                "status": "failed",
                "progress": 0.0,
                "error": describe_error(e)[:300],
            }
            return

    log.info("%s | ok em %.1fs", tag, time.monotonic() - started)
    _prepare_jobs[url] = {"status": "done", "progress": 1.0, "error": None}


@app.post("/prepare-source-async", status_code=202)
def prepare_source_async(body: PrepareSourceBody, background: BackgroundTasks):
    outputs = [u for u in (body.upload_url, body.proxy_upload_url) if u]
    if not outputs:
        raise HTTPException(
            status_code=400, detail="Nenhuma saída pedida (upload_url ou proxy)."
        )
    if not _is_safe_url(body.url) or not all(_is_safe_url(u) for u in outputs):
        raise HTTPException(status_code=400, detail="URL não permitida.")
    job = _prepare_jobs.get(body.url)
    if job and job.get("status") == "processing":
        return {"status": "processing", "progress": job.get("progress", 0.0)}
    if job and job.get("status") == "failed":
        # REPORTA a falha uma vez e esquece o job. Sem isso o chamador nunca vê
        # "failed" (a linha abaixo sobrescreveria o estado) e o polling do export
        # dispararia um ffmpeg novo a cada consulta, indefinidamente. Esquecer o
        # job faz a PRÓXIMA chamada ser uma nova tentativa, que é o que o
        # professor espera ao clicar em exportar de novo.
        _prepare_jobs.pop(body.url, None)
        return {"status": "failed", "progress": 0.0, "error": job.get("error")}
    _prepare_jobs[body.url] = {"status": "processing", "progress": 0.0, "error": None}
    background.add_task(
        _run_prepare,
        body.url,
        body.upload_url,
        body.duration_sec or 0.0,
        body.force_fps,
        body.proxy_upload_url,
        body.proxy_height,
        body.proxy_audio,
    )
    return {"status": "processing", "progress": 0.0}


@app.post("/transcribe-async", status_code=202)
def transcribe_async(body: AsyncBody, background: BackgroundTasks):
    if not CALLBACK_URL:
        raise HTTPException(
            status_code=503,
            detail="TRANSCRIPTION_CALLBACK_URL não configurada.",
        )
    log.info(
        "lesson=%s | job aceito (file_id=%s)", body.lesson_id, body.file_id
    )
    background.add_task(
        process_and_callback,
        body.file_id,
        body.prompt,
        body.lesson_id,
    )
    return {"accepted": True}
