"""Fila da transcrição por URL: jobs em memória, consultados por polling.

POR QUE POLLING (e não callback como o fluxo do Drive): quem chama (o Moovi
Marketing) consulta o estado até terminar. Assim o serviço não precisa saber a
URL de quem chamou (nada de destino de callback vindo na request, que viraria
SSRF) e funciona também com o chamador rodando local, fora da rede do servidor.

O estado vive em memória: reiniciar o serviço perde os jobs e a consulta passa a
responder "não encontrado"; o chamador trata isso como falha e pede de novo.
Jobs terminados ficam guardados por `ttl` segundos para quem consultar depois
(a pessoa pode fechar a tela no meio e voltar mais tarde).

Sem dependência do FastAPI: dá para testar com Python puro.
"""

import logging
import queue
import threading
import time
from typing import Callable

log = logging.getLogger("speech")

# run(params, set_phase) -> resultado (dict). set_phase marca a etapa atual.
Runner = Callable[[dict, Callable[[str], None]], dict]


class TranscribeJobs:
    def __init__(
        self,
        run: Runner,
        concurrency: int = 1,
        max_queue: int = 32,
        ttl: float = 24 * 3600,
    ) -> None:
        self._run = run
        self._ttl = ttl
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []  # ids na fila, em ordem (para a posição)
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._concurrency = max(1, concurrency)
        self._started = False

    def start(self) -> None:
        """Sobe os workers uma vez. Daemon: não seguram o shutdown."""
        if self._started:
            return
        self._started = True
        for n in range(self._concurrency):
            threading.Thread(target=self._worker, args=(n,), daemon=True, name=f"url-job-{n}").start()

    # --- API pública ------------------------------------------------------

    def submit(self, job_id: str, params: dict) -> dict | None:
        """Enfileira o job (idempotente pelo id) e devolve o estado público.
        None = fila cheia (transitório: o chamador tenta de novo)."""
        with self._lock:
            self._purge()
            job = self._jobs.get(job_id)
            if job is not None:
                return self._public(job_id, job)
            job = {
                "status": "queued",
                "phase": None,
                "params": params,
                "created_at": time.time(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
            try:
                self._queue.put_nowait(job_id)
            except queue.Full:
                return None
            self._jobs[job_id] = job
            self._order.append(job_id)
            return self._public(job_id, job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            self._purge()
            job = self._jobs.get(job_id)
            return self._public(job_id, job) if job is not None else None

    def pending(self) -> int:
        with self._lock:
            return len(self._order)

    # --- interno ------------------------------------------------------------

    def _public(self, job_id: str, job: dict) -> dict:
        out = {"job_id": job_id, "status": job["status"], "phase": job["phase"]}
        if job["status"] == "queued":
            out["ahead"] = self._order.index(job_id) if job_id in self._order else 0
        if job["status"] == "done":
            out["result"] = job["result"]
        if job["status"] == "failed":
            out["error"] = job["error"]
        return out

    def _purge(self) -> None:
        now = time.time()
        stale = [
            jid
            for jid, j in self._jobs.items()
            if j["finished_at"] is not None and now - j["finished_at"] > self._ttl
        ]
        for jid in stale:
            self._jobs.pop(jid, None)

    def _set(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _worker(self, n: int) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job_id in self._order:
                        self._order.remove(job_id)
                    if job is None:
                        continue
                    job["status"] = "processing"
                    params = job["params"]
                try:
                    result = self._run(params, lambda phase: self._set(job_id, phase=phase))
                    self._set(job_id, status="done", phase=None, result=result, finished_at=time.time())
                except Exception as e:  # noqa: BLE001 — qualquer falha vira "failed"
                    message = getattr(e, "detail", None) or str(e) or type(e).__name__
                    self._set(
                        job_id,
                        status="failed",
                        error=str(message)[:500],
                        finished_at=time.time(),
                    )
            except Exception:  # noqa: BLE001 — worker não pode morrer
                log.exception("url-job | worker %d falhou no job %s", n, job_id)
            finally:
                self._queue.task_done()
