#!/usr/bin/env python3
"""単語ジェネ - IELTS/アカデミック英単語から例文を生成するStreamlit Web App"""

import base64
import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html

try:
    import stripe  # 課金(任意機能)。未インストール/未設定でもアプリは動く
except Exception:
    stripe = None

from database import (
    delete_sentence,
    get_all_sentences,
    get_audio_blob,
    get_audio_blobs,
    get_audio_ids,
    get_image_data,
    get_image_data_batch,
    get_monthly_generation_count,
    get_monthly_usage,
    get_sentences_by_status,
    get_used_words,
    init_db,
    is_paid,
    mark_judgments_batch,
    mark_status_and_view,
    record_usage,
    save_audio_blob,
    save_image_data,
    save_sentence,
    search_sentences,
    update_sentence_content,
)

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:  # 依存が未導入の環境でもアプリ自体は起動できるようにする
    streamlit_js_eval = None

load_dotenv()

# Streamlit Cloudではst.secrets、ローカルでは.envからAPIキーを読む。
# secrets.tomlが無いローカル環境ではst.secretsへのアクセスが例外を投げるため握りつぶす。
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}

for _key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "UNSPLASH_ACCESS_KEY"):
    if _key in _secrets and not os.getenv(_key):
        os.environ[_key] = _secrets[_key]

OPENAI_TTS_VOICE = "nova"
OPENAI_TTS_MODEL = "tts-1"


@st.cache_data(show_spinner=False)
def _openai_tts(text: str, voice: str = OPENAI_TTS_VOICE) -> bytes:
    """OpenAI TTS で英文を mp3 バイト列に変換。同一入力はキャッシュする(プロセス内)。"""
    from openai import OpenAI

    client = OpenAI()
    response = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return response.content


def get_or_generate_audio(row_id: int, text: str) -> bytes:
    """DBキャッシュ優先で音声バイトを返す。未生成ならOpenAIで作って保存。"""
    cached = get_audio_blob(row_id)
    if cached:
        return cached
    audio = _openai_tts(text)
    save_audio_blob(row_id, audio)
    return audio


@st.cache_data(show_spinner=False)
def _fetch_unsplash_images(word: str, count: int = 5) -> list[dict]:
    """Unsplash API で単語に紐づく画像情報を取得。"""
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
        return []
    url = (
        "https://api.unsplash.com/search/photos?"
        f"query={urllib.parse.quote(word)}&per_page={count}&orientation=landscape"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Client-ID {key}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.load(resp)
    return [
        {
            "thumb": r["urls"]["thumb"],
            "small": r["urls"]["small"],
            "alt": r.get("alt_description") or word,
            "photographer": r["user"]["name"],
            "photographer_url": r["user"]["links"]["html"],
            "image_url": r["links"]["html"],
        }
        for r in data.get("results", [])
    ]


def get_or_fetch_images(row_id: int, word: str) -> list[dict]:
    """DBキャッシュ優先で画像情報を返す。未取得ならUnsplashで検索して保存。"""
    all_data = get_image_data(row_id)
    key = word.lower()
    if key in all_data and all_data[key]:
        return all_data[key]
    try:
        results = _fetch_unsplash_images(word)
    except Exception:
        return []
    if results:
        all_data[key] = results
        save_image_data(row_id, all_data)
    return results

SYSTEM_PROMPT = """あなたはIELTS・アカデミック英語の語彙講師です。Duo 3.0のように、短く覚えやすく、実際に使えて少し人間味のある自然な英語例文を作成します。

ガイドライン:
- 例文は短く覚えやすいことを最優先。1文・12〜15語以内を目安に簡潔にする
- 関係詞や接続詞で複数の節をつなげず、平易な構文(SVO中心)にする
- 場面は仕事・お金・勉強・人間関係・日常など、誰もが「あるある」と感じるリアルで身近なものにする。教科書的に無味乾燥な説明文は避ける
- I / my boss / our professor のように一人称や具体的な人物の視点で書き、困った・疲れた・驚いた等の自然な感情や人間味をひとさじ加えて記憶に残る一文にする(Duo 3.0のトーン)
- ただし指定語がアカデミック・専門的な場合(例: hypothesis, photosynthesis)は無理に日常や恋愛にこじつけず、職場・研究・授業・社会・ニュースなど、その語が実際に使われる自然な場面を選ぶ
- 非現実的・突飛な設定(動物が交渉する等)は避け、現実にありそうな場面に限る
- 指定された単語をすべて文法的に自然な形で組み込む
- 指定語が互いに無関係でも、現実にありそうな1つの場面を想定し、その文脈の中で各語を自然に使う
- 単語名を引用・列挙する文(例: 「AかBかCか議論した」)は禁止。語を詰め込むための無理な構文を避け、ネイティブが実際に書く・話す自然さを最優先する
- 各単語の訳語・意味は、IELTSやアカデミックな文章で問われる主要な意味を優先する。例文での特殊な用法に引きずられず、辞書の中心的な語義を示す
- 和訳は自然な日本語にする
- 解説は必ず指定フォーマットを厳守する(マークダウン記法やサブ箇条書きは使わない)"""


def _parse_response(text: str) -> dict[str, str]:
    english = re.search(r"【英文】\s*(.*?)(?=【和訳】|$)", text, re.DOTALL)
    japanese = re.search(r"【和訳】\s*(.*?)(?=【解説】|$)", text, re.DOTALL)
    explanation = re.search(r"【解説】\s*(.*?)$", text, re.DOTALL)
    return {
        "english": english.group(1).strip() if english else "",
        "japanese": japanese.group(1).strip() if japanese else "",
        "explanation": explanation.group(1).strip() if explanation else "",
    }


def _record_api_usage(model: str, response) -> None:
    """APIレスポンスの usage を取り出して利用量テーブルに記録する。
    記録は付帯処理なので、何が起きても生成本体は止めないよう握りつぶす。"""
    try:
        u = response.usage
        record_usage(
            model=model,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            user_email=_current_user(),
        )
    except Exception:
        pass


def generate_sentence(words: list[str]) -> dict[str, str]:
    client = anthropic.Anthropic()
    words_str = "、".join(words)
    user_message = f"""以下の英単語を全て自然に含む、短く覚えやすいアカデミック英語(IELTSレベル)の例文を1つ作成してください。

単語: {words_str}

以下の形式で出力してください（見出し行はそのまま使用）：

【英文】
（ここに例文。1文・15語以内を目安に、複文を避けて短く覚えやすく）

【和訳】
（ここに日本語訳。指定単語すべてに対応する日本語の箇所を、1単語につき必ず1箇所ずつ《》で囲む。例: tint→《色合い》、peanut→《ピーナッツ》）

【解説】
（指定単語ごとに、必ず以下の形式で1行ずつ。1単語=1行のみ。サブ箇条書き禁止。Markdown(**, __)禁止）
- 単語 【品詞】 /IPA発音記号/ (類義語: word1, word2, word3): 訳語を2〜3個だけ読点区切りで(例: 色合い、色調)。説明文にしない。「〜する動詞」等の品詞名や用法説明は書かない。

ルール:
- 品詞は単語の直後に【】で1つ置く(【動】【名】【形】【副】【前】【接】【代】等)
- IPA発音記号は必須(/.../形式)
- 類義語は2〜3個。同一品詞・近い意味の英単語を選ぶ
- 全部1行に収める。改行禁止
- アスタリスク等の装飾文字禁止
- 和訳では指定単語すべてに対応する日本語表現を必ず1箇所ずつ《》で囲む(囲んだ《》の数=指定単語の数。1つも漏らさない。《》は和訳の中だけで使う)

例(この形式で必ず出力):
- negotiate 【動】 /nɪˈɡoʊʃieɪt/ (類義語: discuss, bargain, mediate): 交渉する、協議する
- deadline 【名】 /ˈdedlaɪn/ (類義語: due date, cutoff, time limit): 締切、期限"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.7,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )
    _record_api_usage("claude-haiku-4-5", response)

    text = next((b.text for b in response.content if b.type == "text"), "")
    parsed = _parse_response(text)
    parsed["raw"] = text
    return parsed


EXTRACTION_PROMPT = """この画像は英語学習アプリのスクリーンショットです。
画像内で「色付き(オレンジ・赤・黄色など)で強調表示されている英単語」を抽出してください。

英語学習アプリでは、覚えるべき重要語が文章中でオレンジや赤などのアクセントカラーで強調表示されることが多いです。
通常の本文(黒・グレー)とは色が違う単語のみが抽出対象です。

判定基準(優先順):
1. オレンジ・赤・黄色など、明らかに本文と色が違う英単語
2. ハイライト・下線・太字で強調された英単語
3. ✕・△マークなど苦手判定マークが付いた英単語
4. 上記がない場合: 画像内のすべての見出し英単語

除外対象:
- UIボタン・タブの文字(Home, Settings など)
- 日本語訳・例文・解説の英単語(あくまで覚えるべき単語のみ)
- アプリ名・ロゴの文字

出力ルール:
- 英単語のみを1行1単語で出力
- 重複は除く
- 余計な説明・前置き・番号は付けない
- 単語は小文字に統一して出力

出力例:
component
negotiate
stakeholder"""


def extract_words_from_image(image_bytes: bytes, media_type: str) -> list[str]:
    client = anthropic.Anthropic()
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )
    _record_api_usage("claude-sonnet-4-5", response)

    text = next((b.text for b in response.content if b.type == "text"), "")
    words = [w.strip().lower() for w in text.splitlines() if w.strip() and re.fullmatch(r"[a-zA-Z\-']+", w.strip())]
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _dedup_words(words: list[str]) -> list[str]:
    """順序を保ったまま重複を除去する。"""
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _parse_word_list(text: str) -> list[str]:
    """貼り付けテキストから英単語を抽出。改行/スペース/カンマ等どの区切りでもOK。

    abceed web の単語一覧を HTML ごとコピーした場合にも対応する。
    HTML なら単語は <p class="name">word</p> に入っているのでそこを優先抽出し、
    無ければタグを除去してから単語を拾う(SVGやクラス名をゴミとして拾わないため)。
    """
    text = text or ""
    if "<" in text and "class=" in text:
        names = re.findall(r'class="name"[^>]*>\s*([^<]+?)\s*<', text)
        text = "\n".join(names) if names else re.sub(r"<[^>]+>", " ", text)
    raw = re.split(r"[^a-zA-Z\-']+", text)
    cleaned = [w.strip("-'").lower() for w in raw]
    return _dedup_words([w for w in cleaned if w])


def _chunk(lst: list, n: int = 3) -> list[list]:
    """リストを先頭から n 個ずつのグループに分割する。"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# ── ページ設定 ────────────────────────────────────────────
st.set_page_config(page_title="単語ジェネ", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

# ローカル動作確認用の認証バイパス。環境変数 WG_LOCAL_NOAUTH=1 の時だけログインを飛ばす。
# 本番(Render)では設定しないので影響なし。コミット時はこのフラグを残しても安全。
_LOCAL_NOAUTH = os.getenv("WG_LOCAL_NOAUTH") == "1"


def _current_user() -> str:
    """ログイン中ユーザーの識別子(メール)。ローカルバイパス時は 'local'。
    カードの所有者・利用量集計のキー。これで「本人のデータだけ」を担保する。"""
    if _LOCAL_NOAUTH:
        return "local"
    return getattr(st.user, "email", "") or ""


# ── 無料/有料の枠(工事②) ─────────────────────────────────────
# 無料ユーザーは月 FREE_MONTHLY_LIMIT 回まで生成可。有料(is_paid)は無制限。
# 数値・価格はここだけ直せば変えられる。価格は工事③(Stripe)の購入ボタン文言にも使う。
FREE_MONTHLY_LIMIT = 10
PRICE_LABEL = "¥500/月"


def _quota() -> dict:
    """現在ユーザーの当月生成枠の状況。{paid, used, limit, remaining}。
    remaining は有料なら None(=無制限)。"""
    user = _current_user()
    paid = is_paid(user)
    used = get_monthly_generation_count(user)
    remaining = None if paid else max(0, FREE_MONTHLY_LIMIT - used)
    return {"paid": paid, "used": used, "limit": FREE_MONTHLY_LIMIT, "remaining": remaining}


def _quota_block_message() -> str:
    """無料枠を使い切った時に出す案内文。"""
    return (
        f"今月の無料生成枠（{FREE_MONTHLY_LIMIT}回）を使い切りました。"
        f"来月リセットされます。今すぐ無制限に使うにはアップグレード（{PRICE_LABEL}）をご検討ください。"
    )


# ── ログインゲート: 未ログインなら本体を出さず止める(Render上で有効) ──
if not _LOCAL_NOAUTH and not st.user.is_logged_in:
    st.title("📚 単語ジェネ")
    st.caption("英単語を入れてIELTS/アカデミックな例文を生成するツールです。")
    st.info("ご利用にはログインが必要です。")
    st.button("Googleでログイン", on_click=st.login, type="primary")
    st.stop()

# ログイン済み: サイドバーにアカウント情報とログアウト
with st.sidebar:
    if _LOCAL_NOAUTH:
        st.caption("ローカル確認モード(認証バイパス)")
    else:
        st.caption(f"ログイン中\n\n{getattr(st.user, 'email', '') or st.user.get('name', '')}")
        st.button("ログアウト", on_click=st.logout, use_container_width=True)

# フラッシュカード学習中はタイトル/概要/タブ帯を隠してカードに集中させる
if st.session_state.get("card_mode_rows") is None:
    st.title("📚 単語ジェネ")
    st.caption("英単語を入れてIELTS/アカデミックな例文を生成。覚えにくい単語をまとめて1文に詰め込んで定着させるためのツール。")
else:
    # タブ帯を隠し、上部余白と要素間の隙間を詰めてスマホ1画面に近づける
    st.markdown(
        "<style>"
        "div.st-key-active_tab{display:none;}"  # 学習中はタブ切替(segmented control)を隠す
        "header[data-testid='stHeader']{display:none;}"
        "div[data-testid='stMainBlockContainer'],section[data-testid='stMain'] .block-container"
        "{padding-top:1.2rem;padding-bottom:1rem;}"
        "div[data-testid='stVerticalBlock']{gap:0.45rem;}"
        # 判断ボタンのスタイル(毎フリップ再注入せず、カード入場時のこの全体再実行で一度だけ入れる)
        "div.st-key-judge_buttons div[data-testid='stHorizontalBlock']{flex-wrap:nowrap;gap:8px;}"
        "div.st-key-judge_buttons div[data-testid='stColumn']{min-width:0;}"
        "div.st-key-mark_mastered button{background:#e8975a !important;border-color:#e8975a !important;color:#fff !important;}"
        "div.st-key-mark_mastered button:hover{background:#dd8a4b !important;border-color:#dd8a4b !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

# ── PWA化: ホーム画面に「アプリっぽく」追加できるよう、manifestとiOS用メタタグを
#    親ドキュメントの<head>へ注入する(Streamlitは<head>を直接編集できないのでJSで)。
#    セッションに1度だけ実行(再実行ごとのiframe生成・重複注入を避ける)。
if not st.session_state.get("_pwa_injected"):
    st_html(
        """
        <script>
        const head = window.parent.document.head;
        const add = (tag, attrs) => {
            const el = window.parent.document.createElement(tag);
            for (const k in attrs) el.setAttribute(k, attrs[k]);
            head.appendChild(el);
        };
        if (!window.parent.document.querySelector('link[rel="manifest"]')) {
            add('link', {rel: 'manifest', href: '/app/static/manifest.json'});
            add('link', {rel: 'apple-touch-icon', href: '/app/static/icon-180.png'});
            add('meta', {name: 'apple-mobile-web-app-capable', content: 'yes'});
            add('meta', {name: 'mobile-web-app-capable', content: 'yes'});
            add('meta', {name: 'apple-mobile-web-app-title', content: '単語ジェネ'});
            add('meta', {name: 'apple-mobile-web-app-status-bar-style', content: 'default'});
            add('meta', {name: 'theme-color', content: '#e8975a'});
        }
        </script>
        """,
        height=0,
    )
    st.session_state._pwa_injected = True

init_db()

# 今月のAPI概算コストをサイドバーに表示(api_usage作成後なので確実にテーブルがある)。
# Anthropic側で月$5のハード上限があるので、$5に達して突然止まる前に気づくための可視化。
_COST_CAP_USD = 5.0
try:
    _usage = get_monthly_usage()
    with st.sidebar:
        st.divider()
        _ratio = min(_usage["cost"] / _COST_CAP_USD, 1.0)
        st.progress(_ratio)
        _line = f"今月の概算: ${_usage['cost']:.2f} / ${_COST_CAP_USD:.0f}（{_usage['calls']}回）"
        if _usage["cost"] >= 4.0:
            st.warning(_line + "  上限に接近")
        else:
            st.caption(_line)
except Exception:
    pass  # 集計失敗(初回など)で本体は止めない

# タブ切り替え。st.tabs は全タブの中身を毎回実行してしまい、Streamlitの
# 「クリックごとに全スクリプト再実行」と相まって重い。選択中のタブだけを
# if で実行し、非アクティブな2タブ(数百行)の再描画を丸ごと省く。
_active_tab = st.segmented_control(
    "タブ",
    ["学習", "生成", "一括取込"],
    default="学習",
    key="active_tab",
    label_visibility="collapsed",
)
if _active_tab is None:  # 何も選ばれていない瞬間は学習にフォールバック
    _active_tab = "学習"

# ── タブ1: 生成 ───────────────────────────────────────────
if _active_tab == "生成":
    st.subheader("英単語を入力して例文を生成")

    with st.expander("📷 画像から単語を抽出"):
        uploaded = st.file_uploader(
            "単語リストなどのスクリーンショットをアップロード",
            type=["png", "jpg", "jpeg", "webp"],
            key="img_upload",
        )
        if uploaded is not None:
            st.image(uploaded, caption="アップロード画像", width=300)
            image_bytes = uploaded.getvalue()
            image_hash = hashlib.md5(image_bytes).hexdigest()

            if st.session_state.get("last_image_hash") != image_hash:
                try:
                    with st.spinner("単語抽出中..."):
                        words_found = extract_words_from_image(image_bytes, uploaded.type)
                    st.session_state.last_image_hash = image_hash
                    st.session_state.extracted_words = words_found
                    st.session_state.word_select = []
                    if not words_found:
                        st.warning("単語を抽出できませんでした。別の画像でお試しください。")
                    else:
                        st.success(f"{len(words_found)} 個の単語を抽出しました")
                except Exception as e:
                    st.error(f"抽出エラー: {e}")

        if st.session_state.get("extracted_words"):

            def _sync_words():
                st.session_state.words_input_area = " ".join(st.session_state.word_select)

            st.multiselect(
                "例文に使う単語を1〜3語選択(下の入力欄に自動反映)",
                options=st.session_state.extracted_words,
                max_selections=3,
                key="word_select",
                on_change=_sync_words,
            )

    words_input = st.text_area(
        "単語をスペース区切りで入力（1〜3語）",
        placeholder="例: negotiate deadline stakeholder",
        height=80,
        key="words_input_area",
    )

    _q = _quota()
    if _q["paid"]:
        st.caption("✓ 無制限プラン")
    else:
        st.caption(f"今月の生成: {_q['used']} / {_q['limit']} 回（無料枠）")

    if st.button("例文を生成", type="primary"):
        words = words_input.strip().split()
        if len(words) < 1 or len(words) > 3:
            st.error("1〜3語を入力してください。")
        elif _q["remaining"] == 0:
            st.error(_quota_block_message())
        else:
            try:
                with st.spinner("生成中..."):
                    result = generate_sentence(words)

                if not result["english"]:
                    raw_text = result.get("raw", "")
                    if raw_text and "【英文】" not in raw_text:
                        st.error(
                            "🚫 この単語では英語例文を生成できませんでした。\n\n"
                            "卑語・スラング・不適切な表現はAIが生成を拒否します。"
                            "一般的な英単語(動詞・名詞・形容詞)を入力してください。"
                        )
                    else:
                        st.warning("レスポンスの解析に失敗しました。再度お試しください。")
                else:
                    st.success("生成完了！")
                    col_left, col_right = st.columns([1, 1])
                    with col_left:
                        st.markdown("#### 【英文】")
                        st.info(result["english"])
                        st.markdown("#### 【和訳】")
                        st.info(result["japanese"].replace("《", "").replace("》", ""))
                    with col_right:
                        st.markdown("#### 【解説】")
                        st.info(result["explanation"])

                    new_id = save_sentence(words, result["english"], result["japanese"], result["explanation"], _current_user())
                    st.session_state.deck_payload = None  # 学習タブのデッキを作り直す
                    st.session_state.deck_init = {"screen": "list"}

                    # 先回り生成: 音声 + 各単語の画像をDBに保存しておく(カード遷移を高速化)
                    with st.spinner("音声・画像を準備中..."):
                        try:
                            audio = _openai_tts(result["english"])
                            save_audio_blob(new_id, audio)
                        except Exception:
                            pass
                        if os.getenv("UNSPLASH_ACCESS_KEY"):
                            for w in words:
                                try:
                                    get_or_fetch_images(new_id, w)
                                except Exception:
                                    pass

                    st.caption("💾 保存しました(音声・画像も準備済み)")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# ── タブ: 一括取込 ─────────────────────────────────────────
if _active_tab == "一括取込":
    st.subheader("単語リストから一括で例文を生成")
    st.caption(
        "abceed の Web版(app.abceed.com)で苦手単語の一覧を開き、選択してコピー → 下の欄に貼り付け。"
        "コピーできない場合は📷からスクショを複数枚アップロードしてもOK。"
    )

    with st.expander("📷 スクショから単語を抽出(コピペできないとき)"):
        bulk_imgs = st.file_uploader(
            "苦手単語一覧のスクショ(複数枚OK)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="bulk_img_upload",
        )
        if bulk_imgs and st.button("画像から抽出して下の欄に追加", key="bulk_extract_btn"):
            collected: list[str] = []
            prog = st.progress(0.0)
            for i, f in enumerate(bulk_imgs, 1):
                try:
                    collected += extract_words_from_image(f.getvalue(), f.type)
                except Exception as e:
                    st.warning(f"{f.name}: 抽出失敗 ({e})")
                prog.progress(i / len(bulk_imgs))
            prog.empty()
            existing = _parse_word_list(st.session_state.get("bulk_words_text", ""))
            merged = _dedup_words(existing + [w.lower() for w in collected])
            st.session_state.bulk_words_text = "\n".join(merged)
            st.success(f"{len(collected)} 語抽出。重複を除いて計 {len(merged)} 語になりました。")

    bulk_text = st.text_area(
        "単語リスト(改行・スペース・カンマ区切りどれでもOK)",
        height=220,
        key="bulk_words_text",
        placeholder="negotiate\ndeadline\nstakeholder\n…",
    )

    words = _parse_word_list(bulk_text)
    used = get_used_words(_current_user())
    new_words = [w for w in words if w not in used]
    skipped = len(words) - len(new_words)
    groups = _chunk(new_words, 3)

    caption = f"認識した単語: {len(words)} 語"
    if skipped:
        caption += f"（うち登録済み {skipped} 語をスキップ → 残り {len(new_words)} 語）"
    caption += f" → 例文 {len(groups)} 文を生成します(並び順に3語ずつ)"
    st.caption(caption)

    if st.button(
        f"{len(groups)} 文をまとめて生成",
        type="primary",
        disabled=not new_words,
        key="bulk_gen_btn",
    ):
        prog = st.progress(0.0)
        status = st.empty()
        ok, ng = 0, 0
        errors: list[str] = []
        _paid = is_paid(_current_user())
        _used = get_monthly_generation_count(_current_user())
        _quota_hit = False
        for i, grp in enumerate(groups, 1):
            if not _paid and _used >= FREE_MONTHLY_LIMIT:
                _quota_hit = True  # 無料枠を使い切ったら以降はスキップ(一括での枠バイパス防止)
                break
            status.write(f"生成中… {i}/{len(groups)}　({', '.join(grp)})")
            try:
                res = generate_sentence(grp)
                _used += 1
                if res["english"]:
                    save_sentence(grp, res["english"], res["japanese"], res["explanation"], _current_user())
                    ok += 1
                else:
                    ng += 1
                    if len(errors) < 3:
                        errors.append(f"{', '.join(grp)} → 解析失敗(不適切判定など)")
            except Exception as e:
                ng += 1
                if len(errors) < 3:
                    errors.append(f"{', '.join(grp)} → {type(e).__name__}: {e}")
            prog.progress(i / len(groups))
            time.sleep(0.4)  # レート制限回避のため少し間隔を空ける
        status.empty()
        prog.empty()
        if ok:
            st.session_state.deck_payload = None  # 学習タブのデッキを作り直す
            st.session_state.deck_init = {"screen": "list"}
        msg = f"完了: {ok} 文を保存しました。"
        if ng:
            msg += f" 生成できなかったのが {ng} 文あります。"
        (st.success if ok else st.error)(msg)
        if _quota_hit:
            st.warning(_quota_block_message())
        if errors:
            st.warning("エラー例(最初の数件):\n\n" + "\n".join(f"- {e}" for e in errors))
        st.caption("音声・画像は「学習」タブで各カードを開いたときに生成されます(一括時はスキップ)。")

    st.divider()
    with st.expander("🔄 既存カードを3語ずつ作り直す"):
        st.caption(
            "今ある全カードの単語を集めて重複を除き、3語ずつの新しい例文に作り直します。"
            "5語などの長い既存例文を短くするための一括メンテナンスです。"
        )
        _existing = get_all_sentences(_current_user())
        _all_words = _dedup_words(
            [
                w.strip().lower()
                for row in reversed(_existing)
                for w in row["words"].split(",")
                if w.strip()
            ]
        )
        _rebuild_groups = _chunk(_all_words, 3)
        st.caption(
            f"現在 {len(_existing)} カード / 単語 {len(_all_words)} 語 "
            f"→ 3語ずつ {len(_rebuild_groups)} 文に作り直します。"
        )
        st.warning(
            "⚠️ 既存カードは全て削除され、わかる/わからない等のステータスはリセットされます。"
            "再生成のAPIコストがかかります。途中で閉じると不完全になります。"
        )
        _confirm = st.checkbox(
            "内容を理解した(既存カード削除・ステータスリセット)", key="rebuild_confirm"
        )
        _q2 = _quota()
        _rebuild_quota_ok = _q2["remaining"] is None or _q2["remaining"] >= len(_rebuild_groups)
        if _rebuild_groups and not _rebuild_quota_ok:
            st.caption(
                f"⚠️ 作り直しには {len(_rebuild_groups)} 回分の生成枠が必要です"
                f"（今月の残り {_q2['remaining']} 回）。{PRICE_LABEL} で無制限に。"
            )
        if st.button(
            f"{len(_rebuild_groups)} 文に作り直す",
            type="primary",
            disabled=not (_confirm and _rebuild_groups and _rebuild_quota_ok),
            key="rebuild_btn",
        ):
            _old_ids = [row["id"] for row in _existing]
            prog = st.progress(0.0)
            status = st.empty()
            ok = 0
            failed_words: list[str] = []
            for i, grp in enumerate(_rebuild_groups, 1):
                status.write(f"生成中… {i}/{len(_rebuild_groups)}　({', '.join(grp)})")
                try:
                    res = generate_sentence(grp)
                    if res["english"]:
                        save_sentence(grp, res["english"], res["japanese"], res["explanation"], _current_user())
                        ok += 1
                    else:
                        failed_words += grp
                except Exception:
                    failed_words += grp
                prog.progress(i / len(_rebuild_groups))
            status.empty()
            prog.empty()
            # 1文でも生成できた場合のみ旧カードを削除(全滅時はデータを守る)
            if ok > 0:
                for oid in _old_ids:
                    delete_sentence(oid, _current_user())
                st.session_state.deck_payload = None  # 学習タブのデッキを作り直す
                st.session_state.deck_init = {"screen": "list"}
                msg = f"完了: {ok} 文に作り直し、旧 {len(_old_ids)} カードを削除しました。"
                if failed_words:
                    msg += f" 生成できなかった単語(必要なら再取込): {', '.join(failed_words)}"
                st.success(msg)
            else:
                st.error("1文も生成できませんでした。旧カードは削除していません。時間をおいて再実行してください。")


@st.cache_data(show_spinner=False)
def _parse_word_pronunciations(explanation: str) -> dict[str, str]:
    """解説テキストから 単語→IPA のマップを抽出する。(単語の後の【品詞】は読み飛ばす)"""
    out: dict[str, str] = {}
    for line in explanation.splitlines():
        m = re.search(r"([a-zA-Z][a-zA-Z\-']*)\s*(?:【[^】]*】)?\s*/([^/]+)/", line)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


@st.cache_data(show_spinner=False)
def _parse_word_synonyms(explanation: str) -> dict[str, list[str]]:
    """解説テキストから 単語→類義語リスト のマップを抽出する。(単語の後の【品詞】は読み飛ばす)"""
    out: dict[str, list[str]] = {}
    for line in explanation.splitlines():
        m = re.search(
            r"([a-zA-Z][a-zA-Z\-']*)\s*(?:【[^】]*】)?\s*/[^/]+/\s*\(類義語:\s*([^)]+)\)",
            line,
        )
        if m:
            word = m.group(1).lower()
            syns = [s.strip() for s in m.group(2).split(",") if s.strip()]
            if syns:
                out[word] = syns
    return out


@st.cache_data(show_spinner=False)
def _parse_word_pos(explanation: str) -> dict[str, str]:
    """解説テキストから 単語→品詞(動/名/形 等) のマップを抽出する。"""
    out: dict[str, str] = {}
    for line in explanation.splitlines():
        clean = re.sub(r"^[-•・*]+\s*", "", line.strip())
        clean = re.sub(r"\*+", "", clean)
        m = re.match(r"([a-zA-Z][a-zA-Z\-']*)\s*【([^】]+)】", clean)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def _strip_ipa(text: str) -> str:
    """解説文中の `word /IPA/ (類義語: ...)` から /IPA/ と類義語ブロックを除去。"""
    text = re.sub(r"([a-zA-Z][a-zA-Z\-']*)\s+/[^/]+/\s*\(類義語:[^)]*\)", r"\1", text)
    text = re.sub(r"([a-zA-Z][a-zA-Z\-']*)\s+/[^/]+/", r"\1", text)
    text = re.sub(r"\*\*+", "", text)
    return text


def _order_words_by_sentence(words: list[str], english: str) -> list[str]:
    """学習対象単語を英文中の初出順に並べ替える(語形変化込み)。
    英文に見つからない単語は元の順序のまま末尾に置く。"""
    suffix = "(s|es|ed|ing|er|est|ly|ies|ied|ier|iest|d)?"
    text = english or ""
    big = len(text) + 1

    def first_pos(w: str) -> int:
        wc = w.strip()
        if not wc:
            return big
        m = re.search(rf"\b{re.escape(wc)}{suffix}\b", text, re.IGNORECASE)
        return m.start() if m else big

    return sorted(words, key=first_pos)  # 安定ソート: 同順位(未検出含む)は元の順を保持


def _highlight_target_words(english: str, words: list[str]) -> str:
    """英文中の学習対象単語(語形変化込み)を赤色強調する。HTML文字列を返す。"""
    clean_words = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    escaped = html.escape(english)
    if not clean_words:
        return escaped
    suffix = "(s|es|ed|ing|er|est|ly|ies|ied|ier|iest|d)?"
    alt = "|".join(re.escape(w) for w in clean_words)
    pattern = re.compile(rf"\b({alt}){suffix}\b", re.IGNORECASE)
    return pattern.sub(
        lambda m: f"<span style='color:#e07b3c; font-weight:700;'>{m.group(0)}</span>",
        escaped,
    )


def _format_explanation_html(explanation: str) -> str:
    """解説を綺麗なHTMLリストに整形(LLM出力のMarkdown混在を吸収)。"""
    items: list[str] = []
    for raw in explanation.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•・*]+\s*", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"\*(.+?)\*", r"\1", line)
        line = re.sub(r"\*\*+", "", line)
        line = re.sub(r"__+", "", line)
        m = re.match(r"^([^:：]{1,80})[:：]\s*(.+)$", line)
        if m:
            head = m.group(1).strip()
            body = m.group(2).strip()
            items.append(
                "<li style='margin-bottom:8px; line-height:1.6;'>"
                f"<span style='font-weight:600;'>{html.escape(head)}</span>"
                f"<span>: {html.escape(body)}</span>"
                "</li>"
            )
        else:
            items.append(
                f"<li style='margin-bottom:8px; line-height:1.6;'>{html.escape(line)}</li>"
            )
    if not items:
        return f"<div style='line-height:1.6;'>{html.escape(explanation)}</div>"
    return f"<ul style='margin:0; padding-left:18px;'>{''.join(items)}</ul>"


@st.cache_data(show_spinner=False)
def _parse_word_meanings(explanation: str) -> dict[str, str]:
    """解説テキストから 単語→意味(コア意味・使い方) のマップを抽出する。

    新形式 `word /IPA/ (類義語: ...): 意味` と
    旧形式 `**word（訳）**: 意味` の両方に対応する。
    """
    out: dict[str, str] = {}
    for raw in explanation.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•・*]+\s*", "", line)  # 先頭の箇条書き記号
        line = re.sub(r"\*+", "", line)  # 太字マーカー(**)
        wm = re.match(r"([a-zA-Z][a-zA-Z\-']*)", line)
        if not wm:
            continue
        # IPA・類義語・品詞ブロックを除いてから、最初のコロン以降を意味とみなす
        rest = re.sub(r"\(類義語:[^)]*\)", "", line)
        rest = re.sub(r"/[^/]+/", "", rest)
        rest = re.sub(r"【[^】]*】", "", rest)
        cm = re.search(r"[:：]\s*(.+)$", rest)
        if cm:
            out[wm.group(1).lower()] = cm.group(1).strip()
    return out


def _shorten_meaning(text: str, limit: int = 40) -> str:
    """意味を短くする。複数文の説明は最初の1文だけ、訳語列挙(句点なし)はそのまま。"""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text
    # 句点の後にまだ続きがある=複数文の説明文 → 最初の1文だけにする
    m = re.match(r"(.+?[。.])\s*\S", text)
    if m and len(m.group(1)) <= limit + 5:
        return m.group(1).strip()
    # 訳語列挙や短文はそのまま(長すぎる時だけ文字数で切る)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _highlight_japanese(japanese: str) -> str:
    """和訳中の《...》で囲まれた対応箇所を英文と同じ赤で強調。マーカーが無ければそのまま。"""
    escaped = html.escape(japanese)
    return re.sub(
        r"《(.+?)》",
        lambda m: f"<span style='color:#e07b3c; font-weight:700;'>{m.group(1)}</span>",
        escaped,
    )


def _speak_button(text: str, auto_play: bool = False) -> None:
    """ブラウザ標準TTSで英文を読み上げるボタンを描画。良質ボイスを自動選択。

    auto_play=True なら iframe ロード時に1回自動再生を試みる。
    """
    safe_text = json.dumps(text)
    auto_block = ""
    if auto_play:
        auto_block = f"""
        (function tryAuto(retry) {{
          const voices = window.speechSynthesis.getVoices();
          if (!voices.length && retry < 20) {{
            setTimeout(() => tryAuto(retry + 1), 100);
            return;
          }}
          const chosen = pickBest();
          const u = new SpeechSynthesisUtterance({safe_text});
          if (chosen) {{
            u.voice = chosen;
            u.lang = chosen.lang;
          }} else {{
            u.lang = 'en-US';
          }}
          u.rate = 0.95;
          u.pitch = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }})(0);
        """
    component_html = f"""
    <div style="margin: 8px 0;">
      <button id="speak-btn" style="
        background: #e8975a; color: white; border: none;
        padding: 12px 20px; border-radius: 8px; font-size: 16px;
        cursor: pointer; font-weight: 700; width: 100%;
      ">🔊 タップして英文を聞く</button>
    </div>
    <script>
      const PREFERRED = [
        'Ava (Premium)', 'Allison (Premium)', 'Samantha (Enhanced)', 'Samantha',
        'Ava (Enhanced)', 'Allison (Enhanced)', 'Karen', 'Daniel',
        'Google US English', 'Microsoft Aria Online', 'Microsoft Jenny Online',
        'Microsoft Ava', 'Premium', 'Enhanced', 'Neural'
      ];

      function pickBest() {{
        const voices = window.speechSynthesis.getVoices();
        const en = voices.filter(v => v.lang && v.lang.toLowerCase().startsWith('en'));
        for (const key of PREFERRED) {{
          const found = en.find(v => v.name && v.name.includes(key));
          if (found) return found;
        }}
        return en.find(v => v.lang === 'en-US') || en[0] || voices[0];
      }}

      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();

      document.getElementById('speak-btn').addEventListener('click', () => {{
        const chosen = pickBest();
        const u = new SpeechSynthesisUtterance({safe_text});
        if (chosen) {{
          u.voice = chosen;
          u.lang = chosen.lang;
        }} else {{
          u.lang = 'en-US';
        }}
        u.rate = 0.95;
        u.pitch = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      }});

      {auto_block}
    </script>
    """
    st_html(component_html, height=60)


def _audio_player(audio_bytes: bytes, auto_play: bool = False) -> None:
    """コンパクトなカスタム音声プレイヤー(▶/⏸・±3秒・シークバー・小さい時間表示・赤系配色)。"""
    b64 = base64.b64encode(audio_bytes).decode()
    autoplay_js = "setTimeout(()=>a.play().catch(()=>{}),120);" if auto_play else ""
    component_html = f"""
    <div style="display:flex;align-items:center;gap:8px;background:#fff5f5;border:1px solid #ffd5d5;
                border-radius:10px;padding:6px 10px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
      <audio id="a" src="data:audio/mpeg;base64,{b64}" preload="auto"></audio>
      <button id="bk" class="c">«3</button>
      <button id="pp" class="c p">▶</button>
      <button id="fw" class="c">3»</button>
      <input id="sk" type="range" min="0" max="100" value="0" step="0.1"
             style="flex:1;accent-color:#e8975a;height:4px;">
      <span id="tm" style="font-size:10px;color:#999;min-width:54px;text-align:right;
                           font-variant-numeric:tabular-nums;">0:00/0:00</span>
    </div>
    <style>
      .c{{background:#e8975a;color:#fff;border:none;border-radius:6px;padding:4px 7px;
          font-size:11px;cursor:pointer;font-weight:700;line-height:1;}}
      .c.p{{padding:9px 18px;font-size:19px;border-radius:8px;}}
      .c:hover{{background:#dd8a4b;}}
    </style>
    <script>
      const a=document.getElementById('a'),pp=document.getElementById('pp'),
            sk=document.getElementById('sk'),tm=document.getElementById('tm');
      const f=s=>{{s=Math.max(0,s|0);return (s/60|0)+':'+String(s%60).padStart(2,'0');}};
      function u(){{if(a.duration)sk.value=a.currentTime/a.duration*100;
                    tm.textContent=f(a.currentTime)+'/'+f(a.duration||0);}}
      a.addEventListener('timeupdate',u);a.addEventListener('loadedmetadata',u);
      a.addEventListener('play',()=>pp.textContent='⏸');
      a.addEventListener('pause',()=>pp.textContent='▶');
      a.addEventListener('ended',()=>pp.textContent='▶');
      pp.onclick=()=>{{a.paused?a.play():a.pause();}};
      document.getElementById('bk').onclick=()=>{{a.currentTime=Math.max(0,a.currentTime-3);}};
      document.getElementById('fw').onclick=()=>{{a.currentTime=Math.min(a.duration||1e9,a.currentTime+3);}};
      sk.oninput=()=>{{if(a.duration)a.currentTime=sk.value/100*a.duration;}};
      {autoplay_js}
    </script>
    """
    st_html(component_html, height=54)


def _image_carousel(images: list[dict]) -> None:
    """画像をスワイプ(横スクロール)で1枚ずつ見れるカルーセル。スマホのスワイプに対応。"""
    cards = ""
    for img in images:
        thumb = html.escape(img["thumb"])
        photog = html.escape(img.get("photographer", ""))
        purl = html.escape(img.get("photographer_url", "#"))
        cards += (
            "<div style='scroll-snap-align:start;flex:0 0 100%;box-sizing:border-box;"
            "text-align:center;padding:0 8px;'>"
            f"<img src='{thumb}' style='width:100%;max-width:360px;height:280px;object-fit:cover;"
            "border-radius:8px;display:block;margin:0 auto;'>"
            "<div style='font-size:10px;color:#999;margin-top:3px;'>📷 "
            f"<a href='{purl}' target='_blank' style='color:#999;'>{photog}</a></div>"
            "</div>"
        )
    carousel = f"""
    <div id="car" style="display:flex;overflow-x:auto;scroll-snap-type:x mandatory;
         gap:0;-webkit-overflow-scrolling:touch;scrollbar-width:none;">{cards}</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-top:5px;">
      <button id="pv" class="nav">◀</button>
      <span style="font-size:10px;color:#bbb;">スワイプ / ◀▶ で切替</span>
      <button id="nx" class="nav">▶</button>
    </div>
    <style>
      #car::-webkit-scrollbar{{display:none;}}
      .nav{{background:#e8975a;color:#fff;border:none;border-radius:6px;
            padding:3px 12px;font-size:13px;cursor:pointer;font-weight:700;}}
      .nav:hover{{background:#dd8a4b;}}
    </style>
    <script>
      const c=document.getElementById('car');
      document.getElementById('pv').onclick=()=>c.scrollBy({{left:-c.clientWidth,behavior:'smooth'}});
      document.getElementById('nx').onclick=()=>c.scrollBy({{left:c.clientWidth,behavior:'smooth'}});
    </script>
    """
    st_html(carousel, height=345)


# ── 学習デッキのクライアントサイド化 ──────────────────────────────────────
# 学習中の頻繁な操作(めくり・詳細表示・音声・画像・わかる/わからない)を1つの
# iframe内でJSだけで完結させ、サーバー往復をゼロにする。判定はlocalStorageに溜め、
# 「戻る/作り直す/削除」やデッキ入退場の節目でまとめてPython側が回収する。
_AUDIO_DIR = Path(__file__).parent / "static" / "audio"
_AUDIO_ON_DISK: set[int] = set()  # プロセス内で書き出し済みのid(再書き出しを避ける)


def _write_audio_files(ids: list[int]) -> None:
    """指定idの音声blobを static/audio/{id}.mp3 へ書き出す。"""
    try:
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        for rid, blob in get_audio_blobs(ids).items():
            try:
                (_AUDIO_DIR / f"{rid}.mp3").write_bytes(blob)
                _AUDIO_ON_DISK.add(rid)
            except Exception:
                pass
    except Exception:
        pass


def _materialize_audio(ids: list[int], priority: list[int]) -> None:
    """音声mp3をディスクへ用意する。表示中・直近のカード(priority)だけ同期的に書き、
    残りはバックグラウンドスレッドで書く。全件base64でiframeに埋めると初回マウントが
    激重になるのを避けつつ、デッキ入場の待ち時間も最小化する狙い。"""
    need = [i for i in ids if i not in _AUDIO_ON_DISK]
    if not need:
        return
    need_set = set(need)
    prio = [i for i in priority if i in need_set]
    rest = [i for i in need if i not in set(prio)]
    if prio:
        _write_audio_files(prio)  # 表示する数枚だけ即書き出し
    if rest:
        # 残りは裏で書く(ユーザーがめくって到達する頃には書き終わっている想定)
        threading.Thread(target=_write_audio_files, args=(rest,), daemon=True).start()


def _build_deck(rows: list[dict], start: int = 0) -> list[dict]:
    """card_mode_rows からクライアント側デッキ用のJSONデータを組み立てる。
    解説のパースは @st.cache_data 済みの関数を使うので使い回しが効く。"""
    ids = [r["id"] for r in rows]
    # 音声URLを出すかは「blobを持つか」で即判定(本体は読まない)。実ファイルの書き出しは
    # 表示する数枚を優先し、残りは裏で用意する。
    # 音声は base64 データURIで埋め込む。Streamlitの静的配信は Content-Type:text/plain +
    # nosniff になりブラウザが <audio> として再生を拒否する(→機械音にフォールバック)ため、
    # 静的ファイルURLは使えない。データURIなら audio/mpeg を自前で持つので確実に鳴る。
    audio_blobs = get_audio_blobs(ids)
    imgs_by_id = get_image_data_batch(ids)
    deck: list[dict] = []
    for r in rows:
        rid = r["id"]
        words_list = [w.strip() for w in r["words"].split(",") if w.strip()]
        words_list = _order_words_by_sentence(words_list, r["english"])
        pron = _parse_word_pronunciations(r["explanation"])
        syn = _parse_word_synonyms(r["explanation"])
        mean = _parse_word_meanings(r["explanation"])
        pos = _parse_word_pos(r["explanation"])
        words = []
        for w in words_list:
            lw = w.lower()
            words.append({
                "w": w,
                "pos": pos.get(lw, ""),
                "ipa": pron.get(lw, ""),
                "syn": ", ".join(syn.get(lw, [])),
                "meaning": _shorten_meaning(mean.get(lw, "")),
            })
        imgs: dict[str, list] = {}
        for lw, lst in (imgs_by_id.get(rid, {}) or {}).items():
            imgs[lw] = [
                {"thumb": x.get("thumb", ""), "name": x.get("photographer", ""),
                 "url": x.get("photographer_url", "#")}
                for x in (lst or [])
            ]
        deck.append({
            "id": rid,
            "eng": _highlight_target_words(r["english"], words_list),
            "jp": _highlight_japanese(r["japanese"]),
            "vc": r.get("view_count", 0),
            "status": r.get("status", "new"),
            "wstr": r["words"],  # 検索用の素の単語文字列
            "audio": (
                "data:audio/mpeg;base64," + base64.b64encode(audio_blobs[rid]).decode()
                if rid in audio_blobs else ""
            ),
            "plain": r["english"],
            "words": words,
            "imgs": imgs,
        })
    return deck


# ── 学習タブ全体(一覧+カード)をクライアントサイド化 ────────────────────────
# 一覧⇄カードの切替・フィルタ・検索・めくり・詳細・音声・画像を、すべて1つの
# iframe内のJSで完結させる。サーバー往復が発生するのは「初回ロード・判定の保存・
# 削除・作り直し」だけ。判定はlocalStorageに溜め、隠しボタン wg_trigger 経由で
# Python側が回収する。カード枚数が数十程度なので全データをクライアントに載せても軽い。
_STUDY_TEMPLATE = r"""
<div id="wgapp" style="font-family:-apple-system,BlinkMacSystemFont,'Hiragino Kaku Gothic ProN',sans-serif;max-width:640px;margin:0 auto;padding:2px;color:#111827;"></div>
<style>
  html,body{margin:0;padding:0;}
  #wgapp *{box-sizing:border-box;}
  .wg-btn{border:none;border-radius:8px;cursor:pointer;font-weight:700;}
  .wg-card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:6px 0;box-shadow:0 2px 10px rgba(0,0,0,.06);}
  .wg-judge{display:flex;gap:8px;}
  .wg-judge button{flex:1;padding:12px 0;font-size:15px;}
  .wg-ng{background:#fff;color:#374151;border:1px solid #d1d5db !important;}
  .wg-ok{background:#e8975a;color:#fff;}
  .wg-reveal{width:100%;padding:11px 0;background:#f3f4f6;color:#374151;font-size:14px;}
  .wg-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;}
  .wg-back{background:none;border:none;color:#6b7280;font-size:13px;cursor:pointer;padding:4px;}
  .wg-prog{color:#6b7280;font-size:12px;}
  .wg-audio{display:flex;align-items:center;gap:8px;background:#fff5f5;border:1px solid #ffd5d5;border-radius:10px;padding:6px 10px;}
  .wg-ac{background:#e8975a;color:#fff;border:none;border-radius:6px;padding:4px 7px;font-size:11px;cursor:pointer;font-weight:700;line-height:1;}
  .wg-ac.p{padding:9px 18px;font-size:19px;border-radius:8px;}
  .wg-sk{flex:1;accent-color:#e8975a;height:4px;}
  .wg-tm{font-size:10px;color:#999;min-width:54px;text-align:right;font-variant-numeric:tabular-nums;}
  .wg-en{font-size:21px;line-height:1.55;font-weight:500;}
  .wg-lbl{color:#999;font-size:11px;margin-bottom:4px;}
  .wg-wrow{margin-bottom:7px;}
  .wg-whead{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;}
  .wg-w{font-weight:700;font-size:16px;color:#111827;}
  .wg-pos{font-size:12px;color:#2563eb;font-weight:600;}
  .wg-ipa,.wg-syn{font-size:13px;color:#4b5563;}
  .wg-mean{font-size:13px;color:#374151;line-height:1.4;}
  .wg-imgword{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0;}
  .wg-imgword button{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:14px;padding:4px 12px;font-size:13px;cursor:pointer;}
  .wg-imgword button.on{background:#e8975a;color:#fff;border-color:#e8975a;}
  .wg-car{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;-webkit-overflow-scrolling:touch;}
  .wg-car::-webkit-scrollbar{display:none;}
  .wg-car>div{scroll-snap-align:start;flex:0 0 100%;text-align:center;padding:0 8px;}
  .wg-car img{width:100%;max-width:360px;height:260px;object-fit:cover;border-radius:8px;}
  .wg-sec{background:#fff;color:#6b7280;border:1px solid #e5e7eb;border-radius:8px;padding:10px 0;width:100%;font-size:14px;margin-top:10px;cursor:pointer;}
  .wg-del{background:none;border:none;color:#b91c1c;font-size:12px;cursor:pointer;margin-top:12px;text-decoration:underline;}
  .wg-chips{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 8px;}
  .wg-chip{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:16px;padding:6px 12px;font-size:13px;cursor:pointer;}
  .wg-chip.on{background:#e8975a;color:#fff;border-color:#e8975a;}
  .wg-search{width:100%;padding:10px 12px;border:1px solid #e5e7eb;border-radius:8px;font-size:14px;margin-bottom:6px;}
  .wg-cap{color:#6b7280;font-size:12px;margin:4px 0 8px;}
  .wg-editbar{display:flex;align-items:center;justify-content:space-between;font-size:13px;color:#374151;margin-bottom:6px;}
  .wg-delsel{background:#fff;border:1px solid #d1d5db;color:#b91c1c;border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer;font-weight:700;}
  .wg-row{display:block;width:100%;text-align:left;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:11px 14px;margin-bottom:6px;font-size:14px;cursor:pointer;color:#111827;}
  .wg-rowbtn:hover{background:#fafafa;}
  .wg-empty{color:#6b7280;font-size:14px;padding:16px 4px;text-align:center;}
</style>
<script>
(function(){
  const DECK = __DECK__;
  const INIT = __INIT__;
  const byId={}; DECK.forEach(c=>byId[c.id]=c);
  const live={};                 // id -> {status, vc} 今セッションの上書き
  const S=id=>(live[id]&&live[id].status)||byId[id].status;
  const V=id=>(live[id]&&live[id].vc!=null)?live[id].vc:(byId[id].vc||0);

  let screen='list', filter='all', search='', edit=false;
  let playlist=[], idx=0, revealed=false, imgWord=null, autoplay=false;
  const sel=new Set();
  const root=document.getElementById('wgapp');

  function fit(){try{const h=Math.ceil(document.body.scrollHeight)+10;const fe=window.frameElement;if(fe){fe.style.height=h+'px';if(fe.parentElement)fe.parentElement.style.height=h+'px';}}catch(e){}}
  function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}

  const PREFERRED=['Ava (Premium)','Samantha (Enhanced)','Samantha','Google US English','Microsoft Aria Online','Premium','Enhanced','Neural'];
  function pickBest(){const v=window.speechSynthesis.getVoices();const en=v.filter(x=>x.lang&&x.lang.toLowerCase().startsWith('en'));for(const k of PREFERRED){const f=en.find(x=>x.name&&x.name.includes(k));if(f)return f;}return en.find(x=>x.lang==='en-US')||en[0]||v[0];}
  function speak(t){const u=new SpeechSynthesisUtterance(t);const c=pickBest();if(c){u.voice=c;u.lang=c.lang;}else{u.lang='en-US';}u.rate=.95;window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}
  try{window.speechSynthesis.getVoices();window.speechSynthesis.onvoiceschanged=()=>window.speechSynthesis.getVoices();}catch(e){}

  function loadJ(){try{return JSON.parse(localStorage.getItem('wg_judgments')||'{}');}catch(e){return {};}}
  function fireTrigger(){const b=window.parent.document.querySelector('.st-key-wg_trigger button');if(b)b.click();}
  function act(type,data){localStorage.setItem('wg_action',JSON.stringify(Object.assign({type:type},data||{})));fireTrigger();}
  function flush(){if(localStorage.getItem('wg_judgments'))act('flush');}  // 判定をバックグラウンドで保存(UIは止めない)

  // ── フィルタ/検索 ──
  function filtered(){
    const q=search.trim().toLowerCase();
    return DECK.filter(c=>{
      if(filter!=='all'&&S(c.id)!==filter)return false;
      if(q){const hay=((c.wstr||'')+' '+(c.plain||'')+' '+(c.jp||'')).toLowerCase();if(hay.indexOf(q)<0)return false;}
      return true;
    });
  }

  // ── カード判定/移動 ──
  function judge(status){
    const c=playlist[idx];const j=loadJ();const e=j[c.id]||{status:status,inc:0};e.status=status;e.inc=(e.inc||0)+1;j[c.id]=e;
    localStorage.setItem('wg_judgments',JSON.stringify(j));
    live[c.id]={status:status,vc:V(c.id)+1};
    if(idx>=playlist.length-1){finished();}else{idx++;revealed=false;imgWord=null;autoplay=true;renderCard();}
  }
  function toList(){screen='list';flush();renderList();}
  function openCard(id){playlist=filtered();idx=Math.max(0,playlist.findIndex(c=>c.id===id));screen='card';revealed=false;imgWord=null;autoplay=true;renderCard();}

  // ── 一覧画面 ──
  function renderList(){
    let h='<div class="wg-chips" id="wgchips">';
    [['all','全て'],['new','🆕 新規'],['review','🔁 復習する'],['mastered','✅ 習得済み']].forEach(p=>{h+='<button data-f="'+p[0]+'" class="wg-chip'+(filter===p[0]?' on':'')+'">'+p[1]+'</button>';});
    h+='</div>';
    h+='<input id="wgsearch" class="wg-search" placeholder="🔎 単語・例文で検索" value="'+esc(search)+'">';
    h+='<div class="wg-cap" id="wgcap"></div>';
    h+='<div class="wg-editbar"><label><input type="checkbox" id="wgedit"'+(edit?' checked':'')+'> 選択して削除</label><span id="wgdelwrap"></span></div>';
    h+='<div id="wgrows"></div>';
    root.innerHTML=h;
    document.getElementById('wgsearch').oninput=function(){search=this.value;renderRows();};
    document.getElementById('wgedit').onchange=function(){edit=this.checked;sel.clear();renderList();};
    root.querySelectorAll('.wg-chip').forEach(b=>b.onclick=()=>{filter=b.getAttribute('data-f');root.querySelectorAll('.wg-chip').forEach(x=>x.classList.toggle('on',x===b));renderRows();});
    renderRows();
  }
  function renderRows(){
    const list=filtered();
    const vcs=list.map(c=>V(c.id));const lap=vcs.length?Math.min.apply(null,vcs):0;const mx=vcs.length?Math.max.apply(null,vcs):0;
    const cap=document.getElementById('wgcap');if(cap)cap.innerHTML='📊 '+list.length+'件 / 周回(最小閲覧): <b>'+lap+'</b>　|　最大 '+mx+'回';
    const dw=document.getElementById('wgdelwrap');
    if(dw)dw.innerHTML=edit?'<button class="wg-delsel" id="wgdelsel">🗑 選択('+sel.size+')を削除</button>':'';
    const ds=document.getElementById('wgdelsel');
    if(ds)ds.onclick=()=>{if(sel.size&&confirm(sel.size+'件を削除しますか?'))act('delete',{ids:Array.from(sel)});};
    const rows=document.getElementById('wgrows');
    if(!list.length){rows.innerHTML='<div class="wg-empty">該当するカードがありません。</div>';fit();return;}
    const ic={'new':'🆕','review':'🔁','mastered':'✅'};
    let h='';
    for(const c of list){
      const icon=ic[S(c.id)]||'🆕';const w=esc(c.words.map(x=>x.w).join(' / '));
      if(edit){h+='<label class="wg-row"><input type="checkbox" data-id="'+c.id+'"'+(sel.has(c.id)?' checked':'')+'> '+icon+'  '+w+'　('+V(c.id)+'回)</label>';}
      else{h+='<button class="wg-row wg-rowbtn" data-id="'+c.id+'">'+icon+'  '+w+'　('+V(c.id)+'回)</button>';}
    }
    rows.innerHTML=h;
    if(edit){rows.querySelectorAll('input[type=checkbox]').forEach(cb=>cb.onchange=function(){const id=+this.getAttribute('data-id');if(this.checked)sel.add(id);else sel.delete(id);renderRows();});}
    else{rows.querySelectorAll('.wg-rowbtn').forEach(b=>b.onclick=()=>openCard(+b.getAttribute('data-id')));}
    fit();
  }

  // ── カード画面 ──
  function finished(){
    root.innerHTML='<div class="wg-card" style="text-align:center;"><div style="font-size:18px;font-weight:700;margin-bottom:12px;">🎉 このセットの最後でした!</div>'
      +'<button class="wg-btn wg-ok" id="wgrestart" style="width:100%;padding:12px 0;margin-bottom:8px;">最初から</button>'
      +'<button class="wg-btn wg-reveal" id="wgback2">← 一覧に戻る</button></div>';
    document.getElementById('wgrestart').onclick=()=>{idx=0;revealed=false;imgWord=null;autoplay=true;renderCard();};
    document.getElementById('wgback2').onclick=toList;
    flush();fit();
  }
  function renderImgs(c){
    const car=document.getElementById('wgcar');if(!car)return;
    const imgs=c.imgs[(imgWord||'').toLowerCase()]||[];let h='';
    for(const im of imgs){h+='<div><img src="'+im.thumb+'"><div style="font-size:10px;color:#999;margin-top:3px;">📷 <a href="'+im.url+'" target="_blank" style="color:#999;">'+esc(im.name)+'</a></div></div>';}
    car.innerHTML=h||'<div style="color:#999;font-size:12px;padding:8px;">画像なし</div>';
    car.querySelectorAll('img').forEach(im=>im.onload=fit);fit();
  }
  function setupAudio(c){
    const a=document.getElementById('wgau'),pp=document.getElementById('wgpp'),sk=document.getElementById('wgsk'),tm=document.getElementById('wgtm');
    const f=s=>{s=Math.max(0,s|0);return (s/60|0)+':'+String(s%60).padStart(2,'0');};
    function u(){if(a.duration)sk.value=a.currentTime/a.duration*100;tm.textContent=f(a.currentTime)+'/'+f(a.duration||0);}
    a.addEventListener('timeupdate',u);a.addEventListener('loadedmetadata',u);
    a.addEventListener('play',()=>pp.textContent='⏸');a.addEventListener('pause',()=>pp.textContent='▶');a.addEventListener('ended',()=>pp.textContent='▶');
    a.addEventListener('error',()=>{try{speak(c.plain);}catch(e){}});
    pp.onclick=()=>{a.paused?a.play():a.pause();};
    document.getElementById('wgbk').onclick=()=>{a.currentTime=Math.max(0,a.currentTime-3);};
    document.getElementById('wgfw').onclick=()=>{a.currentTime=Math.min(a.duration||1e9,a.currentTime+3);};
    sk.oninput=()=>{if(a.duration)a.currentTime=sk.value/100*a.duration;};
    if(autoplay){autoplay=false;setTimeout(()=>a.play().catch(()=>{}),120);}
  }
  function detailHTML(c){
    let wb='';
    for(const w of c.words){
      let p='<span class="wg-w">'+esc(w.w)+'</span>';
      if(w.pos)p+='<span class="wg-pos">【'+esc(w.pos)+'】</span>';
      if(w.ipa)p+='<span class="wg-ipa">/'+esc(w.ipa)+'/</span>';
      if(w.syn)p+='<span class="wg-syn">≈ '+esc(w.syn)+'</span>';
      wb+='<div class="wg-wrow"><div class="wg-whead">'+p+'</div>'+(w.meaning?'<div class="wg-mean">: '+esc(w.meaning)+'</div>':'')+'</div>';
    }
    let h='<div class="wg-card"><div class="wg-lbl">単語</div><div style="margin-bottom:10px;">'+wb+'</div><div class="wg-lbl">和訳</div><div style="font-size:14px;line-height:1.5;">'+c.jp+'</div></div>';
    const iws=c.words.map(w=>w.w).filter(w=>(c.imgs[w.toLowerCase()]||[]).length);
    if(iws.length){
      if(!imgWord||iws.indexOf(imgWord)<0)imgWord=iws[0];
      h+='<div class="wg-lbl" style="margin-top:8px;">🖼️ 単語のイメージ</div><div class="wg-imgword" id="wgiw">';
      for(const w of iws)h+='<button data-w="'+esc(w)+'" class="'+(w===imgWord?'on':'')+'">'+esc(w)+'</button>';
      h+='</div><div class="wg-car" id="wgcar"></div>';
    }
    h+='<button class="wg-sec" id="wgregen">🔄 別の例文で作り直す</button>';
    h+='<div style="text-align:center;"><button class="wg-del" id="wgdel">このカードを削除</button></div>';
    return h;
  }
  function wireDetail(c){
    const rg=document.getElementById('wgregen');if(rg)rg.onclick=()=>act('regen',{id:c.id});
    const dl=document.getElementById('wgdel');if(dl)dl.onclick=()=>{if(confirm('このカードを削除しますか?'))act('delete',{ids:[c.id]});};
    const iw=document.getElementById('wgiw');
    if(iw){iw.querySelectorAll('button').forEach(b=>b.onclick=()=>{imgWord=b.getAttribute('data-w');iw.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));renderImgs(c);});renderImgs(c);}
  }
  // 詳細表示は「下の領域だけ差し替え」で行う。カード全体(=音声プレイヤー)を作り直さないので
  // 再生中の音声は止まらず・最初に戻らず、そのまま流れ続ける。
  function reveal(){revealed=true;const d=document.getElementById('wgdetail');if(!d)return;d.innerHTML=detailHTML(playlist[idx]);wireDetail(playlist[idx]);fit();}
  function renderCard(){
    const c=playlist[idx];if(!c){toList();return;}
    let h='';
    h+='<div class="wg-top"><button class="wg-back" id="wgback">← 一覧に戻る</button>'
      +'<span class="wg-prog">'+(idx+1)+' / '+playlist.length+'　|　閲覧 '+V(c.id)+'回</span></div>';
    h+='<div class="wg-judge"><button class="wg-btn wg-ng" id="wgng">✕ わからない</button><button class="wg-btn wg-ok" id="wgok">✓ わかる</button></div>';
    if(c.audio){
      h+='<div class="wg-audio" style="margin-top:8px;"><audio id="wgau" src="'+c.audio+'" preload="auto"></audio>'
        +'<button class="wg-ac" id="wgbk">«3</button><button class="wg-ac p" id="wgpp">▶</button><button class="wg-ac" id="wgfw">3»</button>'
        +'<input id="wgsk" class="wg-sk" type="range" min="0" max="100" value="0" step="0.1"><span class="wg-tm" id="wgtm">0:00/0:00</span></div>';
    }else{h+='<div style="margin-top:8px;"><button class="wg-btn wg-ok" id="wgtts" style="width:100%;padding:11px 0;">🔊 タップして英文を聞く</button></div>';}
    h+='<div class="wg-card"><div class="wg-lbl">【英文】</div><div class="wg-en">'+c.eng+'</div></div>';
    h+='<div id="wgdetail">'+(revealed?detailHTML(c):'<button class="wg-btn wg-reveal" id="wgrev">詳細を見る</button>')+'</div>';
    root.innerHTML=h;
    document.getElementById('wgback').onclick=toList;
    document.getElementById('wgng').onclick=()=>judge('review');
    document.getElementById('wgok').onclick=()=>judge('mastered');
    if(c.audio){setupAudio(c);}else{const t=document.getElementById('wgtts');if(t)t.onclick=()=>speak(c.plain);}
    if(!revealed){document.getElementById('wgrev').onclick=reveal;}
    else{wireDetail(c);}
    fit();
  }

  // ── 起動: INITで初期画面を決める(作り直し後はそのカードを開く) ──
  if(INIT&&INIT.screen==='card'&&byId[INIT.id]){openCard(INIT.id);if(INIT.revealed){reveal();}}
  else{renderList();}
})();
</script>
"""


def _render_study(deck: list[dict], init: dict) -> None:
    payload = json.dumps(deck, ensure_ascii=False).replace("</", "<\\/")
    init_json = json.dumps(init, ensure_ascii=False).replace("</", "<\\/")
    html_str = _STUDY_TEMPLATE.replace("__DECK__", payload).replace("__INIT__", init_json)
    st_html(html_str, height=480, scrolling=True)



STATUS_LABEL = {"new": "🆕 新規", "review": "🔁 復習する", "mastered": "✅ 習得済み"}
FILTER_LABEL = {"all": "全て", "new": "🆕 新規", "review": "🔁 復習する", "mastered": "✅ 習得済み"}


# ── タブ: 学習(カード一覧・復習) ───────────────────────────────────────────
if _active_tab == "学習":
    # ── 学習タブ(一覧+カード)を1つのJSコンポーネントで回す ──────────────────
    # 一覧⇄カード・フィルタ・検索・めくり・詳細・音声・画像は全部iframe内のJSで完結。
    # サーバー往復は「初回ロード・判定保存(flush)・削除・作り直し」だけ。判定はJSが
    # localStorageに溜め、隠しボタン wg_trigger 経由でここが回収する。
    # 重要: 判定flushでは deck_payload を作り直さない。同じhtmlを返せばStreamlitは
    # iframeを再マウントしないので、裏で保存してもカード位置や画面状態が保たれる。
    st.markdown("<style>div.st-key-wg_trigger{display:none;}</style>", unsafe_allow_html=True)
    if st.button("t", key="wg_trigger"):  # JSから.click()される隠しトリガ
        st.session_state.wg_pending = True
        st.session_state.wg_nonce = st.session_state.get("wg_nonce", 0) + 1

    if st.session_state.get("wg_pending"):
        if streamlit_js_eval is None:
            st.session_state.wg_pending = False
        else:
            _nonce = st.session_state.get("wg_nonce", 0)
            _raw = streamlit_js_eval(
                js_expressions="JSON.stringify({j:localStorage.getItem('wg_judgments'),a:localStorage.getItem('wg_action')})",
                key=f"wg_read_{_nonce}", want_output=True,
            )
            if _raw is not None:  # None の間はまだ取得待ち(次のrerunで届く)
                try:
                    _payload = json.loads(_raw)
                except Exception:
                    _payload = {}
                _rj = _payload.get("j")
                if _rj:
                    try:
                        _jmap = json.loads(_rj)
                    except Exception:
                        _jmap = {}
                    _items = [
                        (int(k), v.get("status", "review"), int(v.get("inc", 1)))
                        for k, v in _jmap.items()
                    ]
                    if _items:
                        mark_judgments_batch(_items)
                _action = None
                _ra = _payload.get("a")
                if _ra:
                    try:
                        _action = json.loads(_ra)
                    except Exception:
                        _action = None
                if _action:
                    _t = _action.get("type")
                    if _t == "delete":
                        _ids = _action.get("ids") or ([_action["id"]] if _action.get("id") else [])
                        for _i in _ids:
                            try:
                                delete_sentence(int(_i), _current_user())
                            except Exception:
                                pass
                        st.session_state.deck_payload = None  # 構成が変わったので作り直す
                        st.session_state.deck_init = {"screen": "list"}
                    elif _t == "regen":
                        _id = int(_action.get("id", 0))
                        _row = next(
                            (r for r in (st.session_state.get("deck_rows") or []) if r["id"] == _id),
                            None,
                        )
                        if _row:
                            with st.spinner("新しい例文を生成中..."):
                                try:
                                    _regen = generate_sentence(_row["words"].split(","))
                                except Exception as e:
                                    _regen = {}
                                    st.error(f"生成エラー: {e}")
                            if _regen.get("english"):
                                update_sentence_content(
                                    _id, _regen["english"], _regen["japanese"], _regen["explanation"]
                                )
                            st.session_state.deck_payload = None
                            st.session_state.deck_init = {"screen": "card", "id": _id, "revealed": True}
                    # "flush" は判定を反映するだけ。画面状態(deck_init)は変えない。
                st.session_state.wg_pending = False
                st.session_state.wg_nonce = _nonce + 1
                streamlit_js_eval(
                    js_expressions="localStorage.removeItem('wg_judgments');localStorage.removeItem('wg_action');",
                    key=f"wg_clr_{_nonce}",
                )

    # デッキデータを構築してセッションにキャッシュする。判定flushでは作り直さないので
    # 同一htmlになり、iframeは再マウントされない(=裏で保存しても画面が飛ばない)。
    if st.session_state.get("deck_payload") is None:
        _rows = get_sentences_by_status(None, _current_user())
        st.session_state.deck_rows = _rows
        st.session_state.deck_payload = _build_deck(_rows, 0)

    if not st.session_state.deck_payload:
        st.info("まだカードがありません。「生成」か「一括取込」タブで作ってください。")
    else:
        _render_study(
            st.session_state.deck_payload,
            st.session_state.get("deck_init", {"screen": "list"}),
        )
