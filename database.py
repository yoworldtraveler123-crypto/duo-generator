"""SQLite操作モジュール"""

from __future__ import annotations  # `bytes | None` 等の注釈を Python 3.9 でも使えるように

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO sentences (created_at, words, english, japanese, explanation) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ",".join(words), english, japanese, explanation),
        )
        return cur.lastrowid


def get_all_sentences() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM sentences ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def search_sentences(keyword: str) -> list[dict]:
    like = f"%{keyword}%"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sentences WHERE english LIKE ? OR japanese LIKE ? OR words LIKE ? ORDER BY created_at DESC",
            (like, like, like),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_sentence(row_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM sentences WHERE id = ?", (row_id,))


def update_status(row_id: int, status: str) -> None:
    if status not in {"new", "review", "mastered"}:
        raise ValueError(f"invalid status: {status}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sentences SET status = ? WHERE id = ?", (status, row_id))


def increment_view_count(row_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sentences SET view_count = view_count + 1 WHERE id = ?", (row_id,))


def get_audio_blob(row_id: int) -> bytes | None:
    """指定IDの音声バイナリ(mp3)を取得。未生成なら None。"""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT audio_blob FROM sentences WHERE id = ?", (row_id,)).fetchone()
    if not row or not row[0]:
        return None
    return bytes(row[0])


def save_audio_blob(row_id: int, audio_bytes: bytes) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sentences SET audio_blob = ? WHERE id = ?", (audio_bytes, row_id))


def update_sentence_content(row_id: int, english: str, japanese: str, explanation: str) -> None:
    """例文・和訳・解説を上書きする。英文が変わるので音声キャッシュ(audio_blob)もクリアする。
    閲覧回数・ステータス・単語リスト・画像は保持する。"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sentences SET english=?, japanese=?, explanation=?, audio_blob=NULL WHERE id=?",
            (english, japanese, explanation, row_id),
        )


def get_image_data(row_id: int) -> dict:
    """指定IDの画像URLマップを取得。未保存なら空辞書。"""
    import json as _json

    with sqlite3.connect(DB_PATH) as conn:
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

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE sentences SET image_data = ? WHERE id = ?", (_json.dumps(data), row_id))


def get_sentences_by_status(status: str | None) -> list[dict]:
    """statusで絞り込んだ履歴を取得。None or 'all' なら全件。"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if status in (None, "all"):
            rows = conn.execute("SELECT * FROM sentences ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sentences WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_used_words() -> set[str]:
    """過去の例文で対象に使われた単語(小文字)の集合を返す。一括取込の重複スキップ用。"""
    with sqlite3.connect(DB_PATH) as conn:
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
