"""Acesso ao banco do moovi_class (MySQL) — SOMENTE as colunas de export.

O render roda aqui, mas quem manda na gravação é o Class: este módulo escreve
apenas `recordings.export_*`, nunca cria linha, nunca altera schema. O Class
continua sendo o dono do modelo (Prisma) e de qualquer DDL.

Conexão curta por operação, de propósito: o progresso é escrito de poucos em
poucos por cento, e uma conexão ociosa presa num servidor remoto morre sozinha
no `wait_timeout` do MySQL — reabrir custa milissegundos e nunca falha por
conexão zumbi.
"""

import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import pymysql

log = logging.getLogger("speech")

# Falha ao escrever progresso NÃO derruba o render: o arquivo final no MinIO é a
# verdade do "pronto", e o Class reconcilia por ele. Só o log denuncia.
_warned_missing = threading.Event()


def _dsn() -> dict | None:
    """Parseia a DATABASE_URL do Class (`mysql://user:senha@host:porta/base`)."""
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        return None
    u = urlparse(raw)
    if not u.hostname or not u.path:
        return None
    return {
        "host": u.hostname,
        "port": u.port or 3306,
        "user": unquote(u.username or ""),
        "password": unquote(u.password or ""),
        "database": u.path.lstrip("/"),
        "charset": "utf8mb4",
        "autocommit": True,
        "connect_timeout": 10,
        "read_timeout": 30,
        "write_timeout": 30,
    }


def configured() -> bool:
    return _dsn() is not None


@contextmanager
def _cursor():
    dsn = _dsn()
    if dsn is None:
        if not _warned_missing.is_set():
            _warned_missing.set()
            log.warning("db | DATABASE_URL ausente — progresso não será gravado")
        yield None
        return
    conn = pymysql.connect(**dsn)
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()


def update_export(
    recording_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    spec_hash: str | None = None,
    exported_at: datetime | None = None,
) -> None:
    """Atualiza as colunas de export da gravação. Erro aqui é logado, não lançado.

    `error` só é gravado quando `status` é FAILED; em qualquer outra transição a
    coluna é LIMPA — senão a mensagem da falha anterior sobreviveria a um
    reexport bem-sucedido e a interface mostraria erro num arquivo que existe.
    """
    sets: list[str] = []
    args: list[object] = []
    if status is not None:
        sets.append("export_status = %s")
        args.append(status)
        sets.append("export_error = %s")
        args.append((error or "")[:500] if status == "FAILED" else None)
    if progress is not None:
        sets.append("export_progress = %s")
        args.append(max(0.0, min(1.0, progress)))
    if spec_hash is not None:
        sets.append("export_spec_hash = %s")
        args.append(spec_hash)
    if exported_at is not None:
        sets.append("exported_at = %s")
        args.append(exported_at.astimezone(timezone.utc).replace(tzinfo=None))
    if not sets:
        return
    sets.append("updated_at = %s")
    args.append(datetime.now(timezone.utc).replace(tzinfo=None))
    args.append(recording_id)
    sql = f"UPDATE recordings SET {', '.join(sets)} WHERE id = %s"
    try:
        with _cursor() as cur:
            if cur is None:
                return
            cur.execute(sql, args)
    except Exception as e:  # noqa: BLE001
        log.warning("db | falha ao gravar export de %s: %s", recording_id, e)


def export_cancelled(recording_id: str) -> bool:
    """A gravação saiu de QUEUED/RENDERING? (o professor cancelou ou reexportou)

    Lido entre os passos do render pra abortar um trabalho que ninguém mais
    espera, em vez de queimar CPU até o fim e sobrescrever o estado novo.
    """
    try:
        with _cursor() as cur:
            if cur is None:
                return False
            cur.execute(
                "SELECT export_status FROM recordings WHERE id = %s", (recording_id,)
            )
            row = cur.fetchone()
    except Exception:  # noqa: BLE001
        return False
    if not row:
        return True
    return row[0] not in ("QUEUED", "RENDERING")
