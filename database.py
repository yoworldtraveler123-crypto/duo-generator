"""DB操作モジュール（SQLite / Turso libSQL 両対応）。

本番(Render)では環境変数 TURSO_DATABASE_URL / TURSO_AUTH_TOKEN があれば
Turso(libSQLクラウド)に接続して永続化する。Renderの無料プランはファイルが
再起動で揮発するため、SQLiteファイルをサーバーに置くと履歴が消える。
環境変数が無いローカル環境では従来通り history.db (SQLite) を使う。
"""

from __future__ import annotations  # `bytes | None` 等の注釈を Python 3.9 でも使えるように

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"
_TURSO_URL = os.getenv("TURSO_DATABASE_URL")
_TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
_USE_TURSO = bool(_TURSO_URL)


@contextmanager
def _connect():
    """DB接続を返す。with を抜ける時に commit し、必ず close する。
    本番は Turso(libSQL)、ローカルは SQLite。SQL文はSQLite互換なので共通。"""
    if _USE_TURSO:
        import libsql_experimental as libsql

        conn = libsql.connect(database=_TURSO_URL, auth_token=_TURSO_TOKEN)
    else:
        conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _dicts(cur) -> list[dict]:
    """カーソルの結果を dict のリストに変換する（row_factory非依存でlibSQLでも動く）。"""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                words       TEXT    NOT NULL,
                english     TEXT    NOT NULL,
                japanese    TEXT    NOT NULL,
                explanation TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'new',
                view_count  INTEGER NOT NULL DEFAULT 0,
                audio_blob  BLOB
            )
        """)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sentences)").fetchall()}
        if "status" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN status TEXT NOT NULL DEFAULT 'new'")
        if "view_count" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN view_count INTEGER NOT NULL DEFAULT 0")
        if "audio_blob" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN audio_blob BLOB")
        if "image_data" not in cols:
            conn.execute("ALTER TABLE sentences ADD COLUMN image_data TEXT")


def save_sentence(words: list[str], english: str, japanese: str, explanation: str) -> int:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sentences (created_at, words, english, japanese, explanation) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ",".join(words), english, japanese, explanation),
        )
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0])


def get_all_sentences() -> list[dict]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM sentences ORDER BY created_at DESC")
        return _dicts(cur)


def search_sentences(keyword: str) -> list[dict]:
    like = f"%{keyword}%"
    with _connect() as conn:
        cur = conn.execute(
            "SELECT * FROM sentences WHERE english LIKE ? OR japanese LIKE ? OR words LIKE ? ORDER BY created_at DESC",
            (like, like, like),
        )
        return _dicts(cur)


def delete_sentence(row_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sentences WHERE id = ?", (row_id,))


def update_status(row_id: int, status: str) -> None:
    if status not in {"new", "review", "mastered"}:
        raise ValueError(f"invalid status: {status}")
    with _connect() as conn:
        conn.execute("UPDATE sentences SET status = ? WHERE id = ?", (status, row_id))


def increment_view_count(row_id: int) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sentences SET view_count = view_count + 1 WHERE id = ?", (row_id,))


def get_audio_blob(row_id: int) -> bytes | None:
    """指定IDの音声バイナリ(mp3)を取得。未生成なら None。"""
    with _connect() as conn:
        row = conn.execute("SELECT audio_blob FROM sentences WHERE id = ?", (row_id,)).fetchone()
    if not row or not row[0]:
        return None
    return bytes(row[0])


def save_audio_blob(row_id: int, audio_bytes: bytes) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sentences SET audio_blob = ? WHERE id = ?", (audio_bytes, row_id))


def update_sentence_content(row_id: int, english: str, japanese: str, explanation: str) -> None:
    """例文・和訳・解説を上書きする。英文が変わるので音声キャッシュ(audio_blob)もクリアする。
    閲覧回数・ステータス・単語リスト・画像は保持する。"""
    with _connect() as conn:
        conn.execute(
            "UPDATE sentences SET english=?, japanese=?, explanation=?, audio_blob=NULL WHERE id=?",
            (english, japanese, explanation, row_id),
        )


def get_image_data(row_id: int) -> dict:
    """指定IDの画像URLマップを取得。未保存なら空辞書。"""
    import json as _json

    with _connect() as conn:
        row = conn.execute("SELECT image_data FROM sentences WHERE id = ?", (row_id,)).fetchone()
    if not row or not row[0]:
        return {}
    try:
        return _json.loads(row[0])
    except (ValueError, TypeError):
        return {}


def save_image_data(row_id: int, data: dict) -> None:
    """単語→画像情報リストのマップをJSONで保存。"""
    import json as _json

    with _connect() as conn:
        conn.execute("UPDATE sentences SET image_data = ? WHERE id = ?", (_json.dumps(data), row_id))


def get_sentences_by_status(status: str | None) -> list[dict]:
    """statusで絞り込んだ履歴を取得。None or 'all' なら全件。"""
    with _connect() as conn:
        if status in (None, "all"):
            cur = conn.execute("SELECT * FROM sentences ORDER BY created_at DESC")
        else:
            cur = conn.execute(
                "SELECT * FROM sentences WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        return _dicts(cur)


def get_used_words() -> set[str]:
    """過去の例文で対象に使われた単語(小文字)の集合を返す。一括取込の重複スキップ用。"""
    with _connect() as conn:
        rows = conn.execute("SELECT words FROM sentences").fetchall()
    used: set[str] = set()
    for (w,) in rows:
        if not w:
            continue
        for token in w.split(","):
            t = token.strip().lower()
            if t:
                used.add(t)
    return used
