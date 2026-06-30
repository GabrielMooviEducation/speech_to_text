import base64
import json
import os
import tempfile

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from faster_whisper import WhisperModel
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from pydantic import BaseModel

app = FastAPI()
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

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


def run_transcription(file_id: str, prompt: str | None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    try:
        download_from_drive(file_id, path)
        segments, _ = model.transcribe(
            path,
            language="pt",
            vad_filter=True,
            initial_prompt=prompt or DEFAULT_PROMPT,
        )
        return " ".join(seg.text.strip() for seg in segments)
    finally:
        os.remove(path)


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
