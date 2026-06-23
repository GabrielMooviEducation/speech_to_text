import tempfile

from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel

app = FastAPI()
model = WhisperModel("small")


@app.post("/transcribe")
def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        segments, _ = model.transcribe(tmp.name)
        text = " ".join(seg.text.strip() for seg in segments)
    return {"text": text}
