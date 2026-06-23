import json
import os
import tempfile

import requests
from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from pydantic import BaseModel

app = FastAPI()
model = WhisperModel("small", device="cpu", compute_type="int8")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
# JSON da service account vem inteiro na env var GOOGLE_SA_JSON.
creds = service_account.Credentials.from_service_account_info(
    json.loads(os.environ["GOOGLE_SA_JSON"]), scopes=SCOPES
)


class Body(BaseModel):
    file_id: str


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


@app.post("/transcribe")
def transcribe(body: Body):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    try:
        download_from_drive(body.file_id, path)
        segments, _ = model.transcribe(path)
        text = " ".join(seg.text.strip() for seg in segments)
    finally:
        os.remove(path)
    return {"text": text}
