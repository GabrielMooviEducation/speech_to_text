import tempfile

from fastapi import FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()
model = WhisperModel("small")


@app.post("/transcribe")
async def transcribe(request: Request):
    data = await request.body()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = model.transcribe(tmp.name)
        text = " ".join(seg.text.strip() for seg in segments)
    return {"text": text}
