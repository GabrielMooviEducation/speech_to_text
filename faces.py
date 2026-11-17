"""Trilha do rosto (reenquadrar a edição do Moovi Marketing).

Gravação horizontal virando Short vertical: a edição recorta uma faixa do
quadro. Esta trilha diz onde está o rosto a cada 0,5 s, e o Marketing move o
recorte para seguir a pessoa (sentada, de pé, mudando de lugar).

Detecção com o classificador Haar de rosto frontal que vem dentro do OpenCV
(sem modelo para baixar): rápido em quadro pequeno e bom para quem fala de
frente para a câmera. Do vídeo, o ffmpeg tira 2 quadros por segundo já
reduzidos a 320 px e em cinza; de cada quadro fica o maior rosto.
"""

from __future__ import annotations

import json
import logging
import subprocess

import cv2
import numpy as np

log = logging.getLogger("speech_to_text")

FACE_FPS = 2
WIDTH = 320
# Rosto menor que isso (fração da largura) é ruído: quadro, cartaz, reflexo.
MIN_FACE = 0.05


def probe_size(url: str) -> tuple[int, int] | None:
    """Largura e altura do vídeo como ele aparece (rotação do celular aplicada)."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
            "-of", "json", url,
        ],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    try:
        stream = (json.loads(proc.stdout.decode("utf-8", "ignore")).get("streams") or [{}])[0]
    except ValueError:
        return None
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if w <= 0 or h <= 0:
        return None
    rotation = 0
    try:
        rotation = int(float((stream.get("tags") or {}).get("rotate") or 0))
    except ValueError:
        pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotation = int(float(side["rotation"]))
            except (TypeError, ValueError):
                pass
    return (h, w) if abs(rotation) % 180 == 90 else (w, h)


_cascade: cv2.CascadeClassifier | None = None


def cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def largest_face(gray: np.ndarray) -> tuple[float, float, float] | None:
    """Centro (x, y) e largura do maior rosto, em fração do quadro."""
    h, w = gray.shape[:2]
    side = max(12, int(w * MIN_FACE))
    found = cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
    if len(found) == 0:
        return None
    x, y, fw, fh = max(found, key=lambda f: int(f[2]) * int(f[3]))
    return (x + fw / 2) / w, (y + fh / 2) / h, fw / w


def track_from_frames(frames) -> dict | None:
    """Trilha a partir de quadros em cinza (iterável), um por 1/FACE_FPS s."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    ws: list[float | None] = []
    for gray in frames:
        face = largest_face(gray)
        xs.append(round(face[0], 3) if face else None)
        ys.append(round(face[1], 3) if face else None)
        ws.append(round(face[2], 3) if face else None)
    if not any(x is not None for x in xs):
        return None
    return {"fps": FACE_FPS, "x": xs, "y": ys, "w": ws}


def detect_faces(url: str) -> dict | None:
    """Uma passada no vídeo da URL. Duas threads: roda depois da transcrição e
    não pode pesar na máquina. Sem rosto nenhum, None."""
    size = probe_size(url)
    if not size:
        log.warning("trilha do rosto: ffprobe não leu o tamanho do vídeo")
        return None
    w, h = size
    out_h = max(2, round(WIDTH * h / w / 2) * 2)
    frame_bytes = WIDTH * out_h
    cv2.setNumThreads(2)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-threads", "2",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", url,
            "-an", "-sn", "-dn",
            "-vf", f"fps={FACE_FPS},scale={WIDTH}:{out_h}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    def frames():
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                return
            yield np.frombuffer(buf, np.uint8).reshape(out_h, WIDTH)

    try:
        track = track_from_frames(frames())
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    return track
