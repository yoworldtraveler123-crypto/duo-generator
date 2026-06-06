"""DB操作モジュール（SQLite / Turso libSQL 両対応）。

本番(Render)では環境変数 TURSO_DATABASE_URL / TURSO_AUTH_TOKEN があれば
Turso を **embedded replica** モードで使う。ローカルに複製ファイル
(turso_replica.db)を置き、読み込みはローカルから即座に行い(通信ゼロ)、
書き込みの後だけ sync() で Turso へ反映する。Renderはファイルが再起動で
揮発するが、起動時に Turso から pull するのでデータは保たれる。

これにより「リモートDBへ毎クエリ往復(Render Oregon ↔ Turso Tokyo)」で
履歴表示が激重になる問題を回避する。環境変数が無いローカル環境では従来通り
history.db (SQLite) を使う。
"""

from __future__ import annotations  # `bytes | None` 等の注釈を Python 3.9 でも使えるように

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"
# Turso embedded replica のローカル複製ファイル。Renderでは揮発するが起動時にpullし直す。
_REPLICA_PATH = Path(__file__).parent / "turso_replica.db"

# 一覧表示用のカラム。重い audio_blob(mp3バイナリ) と image_data(JSON) は除外する。
# これらは個別に get_audio_blob(id) / get_image_data(id) でカードを開いた時だけ取得する。
_LIST_COLS = "id, created_at, words, english, japanese, explanation, status, view_count"
_TURSO_URL = os.getenv("TURSO_DATABASE_URL")
_TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
_USE_TURSO = bool(_TURSO_URL)

# init_db() は Streamlit の再実行ごとに呼ばれるが、起動時の pull 同期は一度だけにする。
_initialized = False

# バックグラウンド sync の調停用。書き込み後の Turso への push を UI から切り離す。
_sync_state_lock = threading.Lock()
_syncing = False       # 現在 sync スレッドが走っているか
_sync_again = False    # sync 中に新しい書き込みが来たら、終わった後もう一度回す


def _sync_worker() -> None:
    """ローカル複製の変更を Turso へ push する。専用スレッドで実行(UIをブロックしない)。
    自前のコネクションをこのスレッド内で作って使うのでスレッド安全。"""
    global _syncing, _sync_again
    import libsql_experimental as libsql

    while True:
        try:
            conn = libsql.connect(
                str(_REPLICA_PATH), sync_url=_TURSO_URL, auth_token=_TURSO_TOKEN
            )
            try:
                conn.sync()
            finally:
                conn.close()
        except Exception:
            pass  # ネットワーク失敗等。次の書き込み時に再度試みられる
        with _sync_state_lock:
            if _sync_again:
                _sync_again = False
                continue  # 走行中に来た書き込みをまとめて反映
            _syncing = False
            return


def _trigger_sync() -> None:
    """書き込み後に Turso への push をバックグラウンドで起動する。
    既に sync 中なら二重起動せず、終了後にもう一度回すフラグを立てる(コアレス)。"""
    global _syncing, _sync_again
    if not _USE_TURSO:
        return
    with _sync_state_lock:
        if _syncing:
            _sync_again = True
            return
        _syncing = True
    threading.Thread(target=_sync_worker, daemon=True).start()


@contextmanager
def _connect(sync: bool = False):
    """DB接続を返す。with を抜ける時に commit し、必ず close する。
    本番は Turso の embedded replica(ローカル複製)、ローカル開発は SQLite。
    sync=True の書き込みは、close 後に **バックグラウンドで** Turso へ push する
    (同期完了を待たないのでボタン操作が即座に返る)。読み込みでは push しない。"""
    if _USE_TURSO:
        import libsql_experimental as libsql

        conn = libsql.connect(
            str(_REPLICA_PATH), sync_url=_TURSO_URL, auth_token=_TURSO_TOKEN
        )
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
    if sync and _USE_TURSO:
        _trigger_sync()  # ローカルは保存済み。Turso への反映は後追い(非ブロッキング)


def _dicts(cur) -> list[dict]:
    """カーソルの結果を dict のリストに変換する（row_factory非依存でlibSQLでも動く）。"""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def init_db() -> None:
    global _initialized
    # Streamlitは再実行のたびにこれを呼ぶ。Turso利用時は起動時の1回だけ実処理し、
    # 以降は即returnする(毎回 pull/push すると逆に重くなるため)。
    if _USE_TURSO and _initialized:
        return
    # sync=True: スキーマ作成/変更を最後に Turso へ push する
    with _connect(sync=_USE_TURSO) as conn:
        if _USE_TURSO:
            conn.sync()  # 先に Turso から最新を pull してローカル複製を最新化
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
    _initialized = True


def save_sentence(words: list[str], english: str, japanese: str, explanation: str) -> int:
    with _connect(sync=True) as conn:
        conn.execute(
            "INSERT INTO sentences (created_at, words, english, japanese, explanation) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ",".join(words), english, japanese, explanation),
        )
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0])


def get_all_sentences() -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(f"SELECT {_LIST_COLS} FROM sentences ORDER BY created_at DESC")
        return _dicts(cur)


def search_sentences(keyword: str) -> list[dict]:
    like = f"%{keyword}%"
    with _connect() as conn:
        cur = conn.execute(
            f"SELECT {_LIST_COLS} FROM sentences WHERE english LIKE ? OR japanese LIKE ? OR words LIKE ? ORDER BY created_at DESC",
            (like, like, like),
        )
        return _dicts(cur)


def delete_sentence(row_id: int) -> None:
    with _connect(sync=True) as conn:
        conn.execute("DELETE FROM sentences WHERE id = ?", (row_id,))


def update_status(row_id: int, status: str) -> None:
    if status not in {"new", "review", "mastered"}:
        raise ValueError(f"invalid status: {status}")
    with _connect(sync=True) as conn:
        conn.execute("UPDATE sentences SET status = ? WHERE id = ?", (status, row_id))


def increment_view_count(row_id: int) -> None:
    with _connect(sync=True) as conn:
        conn.execute("UPDATE sentences SET view_count = view_count + 1 WHERE id = ?", (row_id,))


def mark_status_and_view(row_id: int, status: str) -> None:
    """ステータス更新と閲覧回数+1を1接続・1syncでまとめて行う(カードめくりの高速化)。
    従来は update_status + increment_view_count で Turso へ2回pushしていた。"""
    if status not in {"new", "review", "mastered"}:
        raise ValueError(f"invalid status: {status}")
    with _connect(sync=True) as conn:
        conn.execute(
            "UPDATE sentences SET status = ?, view_count = view_count + 1 WHERE id = ?",
            (status, row_id),
        )


def mark_judgments_batch(items: list[tuple[int, str, int]]) -> None:
    """カード学習中に溜めた「わかる/わからない」判定をまとめてDBに反映する。
    items は (row_id, status, inc) のリスト。inc はそのカードを判定した回数(=view_count加算分)。
    1接続・1syncでまとめて書くことで、Tursoへのpushを学習セッションあたり1回に抑える。
    めくり毎にサーバー往復していた従来方式を置き換える。"""
    valid = [(i, s, n) for (i, s, n) in items if s in {"new", "review", "mastered"} and n > 0]
    if not valid:
        return
    with _connect(sync=True) as conn:
        for row_id, status, inc in valid:
            conn.execute(
                "UPDATE sentences SET status = ?, view_count = view_count + ? WHERE id = ?",
                (status, inc, row_id),
            )


def get_audio_blobs(ids: list[int]) -> dict[int, bytes]:
    """指定ID群の音声バイナリを一括取得。audio_blobがあるものだけ {id: bytes} で返す。
    デッキ構築時にmp3を静的ファイルへ書き出す用途。1接続でまとめて読む。"""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    out: dict[int, bytes] = {}
    with _connect() as conn:
        cur = conn.execute(
            f"SELECT id, audio_blob FROM sentences WHERE id IN ({placeholders}) AND audio_blob IS NOT NULL",
            tuple(ids),
        )
        for rid, blob in cur.fetchall():
            if blob:
                out[int(rid)] = bytes(blob)
    return out


def get_audio_ids(ids: list[int]) -> set[int]:
    """指定ID群のうち音声blobを持つidの集合を返す(blob本体は読まない=軽い)。
    デッキ構築時に「音声URLを出すか/ブラウザTTSにするか」を即決するため。"""
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id FROM sentences WHERE id IN ({placeholders}) AND audio_blob IS NOT NULL",
            tuple(ids),
        ).fetchall()
    return {int(r[0]) for r in rows}


def get_image_data_batch(ids: list[int]) -> dict[int, dict]:
    """指定ID群の画像URLマップを一括取得。{id: {word: [画像情報,...]}}。
    APIは叩かず保存済みのものだけ返す(未取得の単語は空)。"""
    import json as _json

    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    out: dict[int, dict] = {}
    with _connect() as conn:
        cur = conn.execute(
            f"SELECT id, image_data FROM sentences WHERE id IN ({placeholders}) AND image_data IS NOT NULL",
            tuple(ids),
        )
        for rid, data in cur.fetchall():
            if not data:
                continue
            try:
                out[int(rid)] = _json.loads(data)
            except (ValueError, TypeError):
                pass
    return out


def get_audio_blob(row_id: int) -> bytes | None:
    """指定IDの音声バイナリ(mp3)を取得。未生成なら None。"""
    with _connect() as conn:
        row = conn.execute("SELECT audio_blob FROM sentences WHERE id = ?", (row_id,)).fetchone()
    if not row or not row[0]:
        return None
    return bytes(row[0])


def save_audio_blob(row_id: int, audio_bytes: bytes) -> None:
    with _connect(sync=True) as conn:
        conn.execute("UPDATE sentences SET audio_blob = ? WHERE id = ?", (audio_bytes, row_id))


def update_sentence_content(row_id: int, english: str, japanese: str, explanation: str) -> None:
    """例文・和訳・解説を上書きする。英文が変わるので音声キャッシュ(audio_blob)もクリアする。
    閲覧回数・ステータス・単語リスト・画像は保持する。"""
    with _connect(sync=True) as conn:
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

    with _connect(sync=True) as conn:
        conn.execute("UPDATE sentences SET image_data = ? WHERE id = ?", (_json.dumps(data), row_id))


def get_sentences_by_status(status: str | None) -> list[dict]:
    """statusで絞り込んだ履歴を取得。None or 'all' なら全件。"""
    with _connect() as conn:
        if status in (None, "all"):
            cur = conn.execute(f"SELECT {_LIST_COLS} FROM sentences ORDER BY created_at DESC")
        else:
            cur = conn.execute(
                f"SELECT {_LIST_COLS} FROM sentences WHERE status = ? ORDER BY created_at DESC",
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
