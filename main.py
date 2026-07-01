import base64
import json
import os
import subprocess
import tempfile

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from pydantic import BaseModel

load_dotenv()  # carrega variáveis do arquivo .env, se existir

app = FastAPI()

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

        _local_model = WhisperModel(
            "large-v3", device="cpu", compute_type="int8"
        )
    return _local_model


# Segredo compartilhado com o moovi_class: vai no header do callback pra que o
# receptor confirme que a chamada veio mesmo deste serviço.
CALLBACK_SECRET = os.getenv("TRANSCRIPTION_CALLBACK_SECRET")

# URL FIXA do callback no moovi_class. Definida por env do PRÓPRIO serviço — NÃO
# vem na request — pra não virar SSRF nem oráculo de exfiltração do secret (um
# chamador não pode redirecionar o callback/secret pra um host arbitrário).
CALLBACK_URL = os.getenv("TRANSCRIPTION_CALLBACK_URL")


def load_sa_info():
    raw = os.environ["GOOGLE_SA_JSON"].strip()
    try:
        return json.loads(raw)  # JSON cru
    except json.JSONDecodeError:
        return json.loads(base64.b64decode(raw))  # base64 do JSON


creds = service_account.Credentials.from_service_account_info(
    load_sa_info(), scopes=SCOPES
)


DEFAULT_PROMPT = os.getenv(
    "DEFAULT_PROMPT",
    "Transcrição de um vídeo em português do Brasil.",
)


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


def download_from_drive(file_id: str, dest: str):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    params = {"alt": "media", "supportsAllDrives": "true"}
    with requests.get(url, headers=headers, params=params, stream=True) as r:
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Drive recusou ({r.status_code}): {r.text[:500]}",
            )
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


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
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg falhou: {proc.stderr.decode('utf-8', 'ignore')[:500]}",
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
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI recusou ({resp.status_code}): {resp.text[:500]}",
        )
    return resp.json()["text"].strip()


def run_transcription(file_id: str, prompt: str | None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    audio_path = None
    try:
        download_from_drive(file_id, video_path)
        audio_path = extract_audio(video_path)
        if USE_OPENAI:
            return transcribe_openai(audio_path, prompt)
        return transcribe_local(audio_path, prompt)
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
    try:
        text = run_transcription(file_id, prompt)
        payload = {"lesson_id": lesson_id, "text": text}
    except Exception as e:  # noqa: BLE001 — qualquer falha vira status FAILED
        payload = {"lesson_id": lesson_id, "error": str(e)[:500]}

    headers = {"Content-Type": "application/json"}
    if CALLBACK_SECRET:
        headers["X-Callback-Secret"] = CALLBACK_SECRET
    try:
        requests.post(CALLBACK_URL, json=payload, headers=headers, timeout=30)
    except requests.RequestException:
        pass  # callback inalcançável: o moovi_class fica em PROCESSING (reprocessável)


@app.post("/transcribe-async", status_code=202)
def transcribe_async(body: AsyncBody, background: BackgroundTasks):
    if not CALLBACK_URL:
        raise HTTPException(
            status_code=503,
            detail="TRANSCRIPTION_CALLBACK_URL não configurada.",
        )
    background.add_task(
        process_and_callback,
        body.file_id,
        body.prompt,
        body.lesson_id,
    )
    return {"accepted": True}
