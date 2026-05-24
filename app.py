#!/usr/bin/env python3
"""単語ジェネ - ビジネス英単語から例文を生成するStreamlit Web App"""

import base64
import hashlib
import html
import json
import os
import re
import urllib.parse
import urllib.request

import anthropic
import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html

from database import (
    delete_sentence,
    get_all_sentences,
    get_audio_blob,
    get_image_data,
    get_sentences_by_status,
    get_used_words,
    increment_view_count,
    init_db,
    save_audio_blob,
    save_image_data,
    save_sentence,
    search_sentences,
    update_status,
)

load_dotenv()

if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "UNSPLASH_ACCESS_KEY" in st.secrets and not os.getenv("UNSPLASH_ACCESS_KEY"):
    os.environ["UNSPLASH_ACCESS_KEY"] = st.secrets["UNSPLASH_ACCESS_KEY"]

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

SYSTEM_PROMPT = """あなたはビジネス英語の熟練講師です。TOEIC800点以上レベルの自然なビジネス英語例文を作成します。

ガイドライン:
- 実際のビジネスシーンで使われる自然な英語を使用する
- 指定された単語をすべて文法的に自然な形で組み込む
- 例文はメール、会議、プレゼンテーション等のビジネス場面を想定する
- 和訳は自然な日本語ビジネス表現にする
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


def generate_sentence(words: list[str]) -> dict[str, str]:
    client = anthropic.Anthropic()
    words_str = "、".join(words)
    user_message = f"""以下の英単語を全て自然に含むビジネス英語の例文を1つ作成してください。

単語: {words_str}

以下の形式で出力してください（見出し行はそのまま使用）：

【英文】
（ここに例文）

【和訳】
（ここに日本語訳）

【解説】
（指定単語ごとに、必ず以下の形式で1行ずつ。1単語=1行のみ。サブ箇条書き禁止。Markdown(**, __)禁止）
- 単語 /IPA発音記号/ (類義語: word1, word2, word3): コア意味。ビジネス文脈での使い方。

ルール:
- IPA発音記号は必須(/.../形式)
- 類義語は2〜3個。同一品詞・近い意味のビジネス英単語を選ぶ
- 全部1行に収める。改行禁止
- アスタリスク等の装飾文字禁止

例(この形式で必ず出力):
- negotiate /nɪˈɡoʊʃieɪt/ (類義語: discuss, bargain, mediate): 交渉する。商談や条件調整で使う基本動詞。
- deadline /ˈdedlaɪn/ (類義語: due date, cutoff, time limit): 締切。タスク完了の最終期限を示す。"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0.7,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )

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


def _chunk(lst: list, n: int = 5) -> list[list]:
    """リストを先頭から n 個ずつのグループに分割する。"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# ── ページ設定 ────────────────────────────────────────────
st.set_page_config(page_title="単語ジェネ", page_icon="📚", layout="wide")
st.title("📚 単語ジェネ")
st.caption("ビジネス英単語を入れて例文を生成。覚えにくい単語をまとめて1文に詰め込んで定着させるためのツール。")

init_db()

tab_gen, tab_bulk, tab_hist = st.tabs(["生成", "一括取込", "履歴"])

# ── タブ1: 生成 ───────────────────────────────────────────
with tab_gen:
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
                "例文に使う単語を1〜5語選択(下の入力欄に自動反映)",
                options=st.session_state.extracted_words,
                max_selections=5,
                key="word_select",
                on_change=_sync_words,
            )

    words_input = st.text_area(
        "単語をスペース区切りで入力（1〜5語）",
        placeholder="例: negotiate deadline stakeholder",
        height=80,
        key="words_input_area",
    )

    if st.button("例文を生成", type="primary"):
        words = words_input.strip().split()
        if len(words) < 1 or len(words) > 5:
            st.error("1〜5語を入力してください。")
        else:
            try:
                with st.spinner("生成中..."):
                    result = generate_sentence(words)

                if not result["english"]:
                    raw_text = result.get("raw", "")
                    if raw_text and "【英文】" not in raw_text:
                        st.error(
                            "🚫 この単語ではビジネス英語例文を生成できませんでした。\n\n"
                            "卑語・スラング・不適切な表現はAIが生成を拒否します。"
                            "ビジネスシーンで使う一般的な英単語(動詞・名詞・形容詞)を入力してください。"
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
                        st.info(result["japanese"])
                    with col_right:
                        st.markdown("#### 【解説】")
                        st.info(result["explanation"])

                    new_id = save_sentence(words, result["english"], result["japanese"], result["explanation"])

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

                    st.caption("💾 履歴に保存しました(音声・画像も準備済み)")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# ── タブ: 一括取込 ─────────────────────────────────────────
with tab_bulk:
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
    used = get_used_words()
    new_words = [w for w in words if w not in used]
    skipped = len(words) - len(new_words)
    groups = _chunk(new_words, 5)

    caption = f"認識した単語: {len(words)} 語"
    if skipped:
        caption += f"（うち履歴済み {skipped} 語をスキップ → 残り {len(new_words)} 語）"
    caption += f" → 例文 {len(groups)} 文を生成します(並び順に5語ずつ)"
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
        for i, grp in enumerate(groups, 1):
            status.write(f"生成中… {i}/{len(groups)}　({', '.join(grp)})")
            try:
                res = generate_sentence(grp)
                if res["english"]:
                    save_sentence(grp, res["english"], res["japanese"], res["explanation"])
                    ok += 1
                else:
                    ng += 1
            except Exception:
                ng += 1
            prog.progress(i / len(groups))
        status.empty()
        prog.empty()
        msg = f"完了: {ok} 文を履歴に保存しました。"
        if ng:
            msg += f" 生成できなかったのが {ng} 文あります(不適切判定など)。"
        st.success(msg)
        st.caption("音声・画像は「履歴」タブで各カードを開いたときに生成されます(一括時はスキップ)。")

def _parse_word_pronunciations(explanation: str) -> dict[str, str]:
    """解説テキストから 単語→IPA のマップを抽出する。"""
    out: dict[str, str] = {}
    for line in explanation.splitlines():
        m = re.search(r"([a-zA-Z][a-zA-Z\-']*)\s*/([^/]+)/", line)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def _parse_word_synonyms(explanation: str) -> dict[str, list[str]]:
    """解説テキストから 単語→類義語リスト のマップを抽出する。"""
    out: dict[str, list[str]] = {}
    for line in explanation.splitlines():
        m = re.search(r"([a-zA-Z][a-zA-Z\-']*)\s*/[^/]+/\s*\(類義語:\s*([^)]+)\)", line)
        if m:
            word = m.group(1).lower()
            syns = [s.strip() for s in m.group(2).split(",") if s.strip()]
            if syns:
                out[word] = syns
    return out


def _strip_ipa(text: str) -> str:
    """解説文中の `word /IPA/ (類義語: ...)` から /IPA/ と類義語ブロックを除去。"""
    text = re.sub(r"([a-zA-Z][a-zA-Z\-']*)\s+/[^/]+/\s*\(類義語:[^)]*\)", r"\1", text)
    text = re.sub(r"([a-zA-Z][a-zA-Z\-']*)\s+/[^/]+/", r"\1", text)
    text = re.sub(r"\*\*+", "", text)
    return text


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
        lambda m: f"<span style='color:#ff4b4b; font-weight:700;'>{m.group(0)}</span>",
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
        background: #ff4b4b; color: white; border: none;
        padding: 8px 16px; border-radius: 6px; font-size: 14px;
        cursor: pointer; font-weight: 600;
      ">🔊 英文を聞く</button>
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
    st_html(component_html, height=50)


STATUS_LABEL = {"new": "🆕 新規", "review": "🔁 復習する", "mastered": "✅ 習得済み"}
FILTER_LABEL = {"all": "全て", "new": "🆕 新規", "review": "🔁 復習する", "mastered": "✅ 習得済み"}


# ── タブ2: 履歴 ───────────────────────────────────────────
with tab_hist:
    filter_choice = st.radio(
        "表示するカード",
        options=list(FILTER_LABEL.keys()),
        format_func=lambda k: FILTER_LABEL[k],
        horizontal=True,
        key="status_filter",
    )

    search_query = st.text_input(
        "キーワード検索（単語から）",
        placeholder="例: negotiate",
        key="hist_search",
    )

    if search_query:
        base_rows = search_sentences(search_query)
        if filter_choice != "all":
            base_rows = [r for r in base_rows if r.get("status") == filter_choice]
        rows = base_rows
    else:
        rows = get_sentences_by_status(filter_choice if filter_choice != "all" else None)

    if rows:
        view_counts = [r.get("view_count", 0) for r in rows]
        lap_count = min(view_counts) if view_counts else 0
        st.caption(
            f"📊 {len(rows)}件 / 周回数(最小閲覧回数): **{lap_count}**　|　最大閲覧: {max(view_counts) if view_counts else 0}回"
        )

    if not rows:
        st.info("該当するカードがありません。" if (search_query or filter_choice != "all") else "履歴がありません。")
        st.session_state.pop("card_mode_rows", None)
    elif st.session_state.get("card_mode_rows") is not None:
        # ── フラッシュカードモード ──
        card_rows = st.session_state.card_mode_rows
        idx = st.session_state.get("card_index", 0)
        idx = max(0, min(idx, len(card_rows) - 1))
        row = card_rows[idx]

        # 閲覧回数の自動カウント + カード変更時は詳細表示をリセット + 音声自動再生フラグを立てる
        if st.session_state.get("last_viewed_card_id") != row["id"]:
            increment_view_count(row["id"])
            row["view_count"] = row.get("view_count", 0) + 1
            st.session_state.last_viewed_card_id = row["id"]
            st.session_state.card_revealed = False
            st.session_state.autoplay_pending = True

        col_back, col_count = st.columns([1, 2])
        with col_back:
            if st.button("← 一覧に戻る", key="back_to_list"):
                st.session_state.pop("card_mode_rows", None)
                st.session_state.pop("card_index", None)
                st.session_state.pop("last_viewed_card_id", None)
                st.rerun()
        with col_count:
            st.markdown(
                f"<div style='text-align:right; padding-top:8px; color:#666;'>"
                f"{idx + 1} / {len(card_rows)}　|　閲覧 {row.get('view_count', 0)}回"
                f"</div>",
                unsafe_allow_html=True,
            )

        words_list = [w.strip() for w in row["words"].split(",") if w.strip()]
        highlighted_english = _highlight_target_words(row["english"], words_list)

        # ── 音声プレイヤー(fragmentの外: 詳細トグル時に再描画させない) ──
        should_autoplay = st.session_state.pop("autoplay_pending", False)
        try:
            audio_bytes = get_or_generate_audio(row["id"], row["english"])
            st.audio(audio_bytes, format="audio/mp3", autoplay=should_autoplay)
        except Exception as e:
            st.warning(f"OpenAI TTS失敗、ブラウザTTSにフォールバック: {e}")
            _speak_button(row["english"], auto_play=should_autoplay)

        # ── 表/裏切替はfragment内に閉じ込めて、外側の音声プレイヤーを再描画しない ──
        @st.fragment
        def _reveal_section():
            revealed = st.session_state.get("card_revealed", False)

            if not revealed:
                st.markdown(
                    f"""
                    <div style='
                        background: #fff; border: 1px solid #e5e7eb;
                        border-radius: 12px; padding: 24px; margin: 16px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                    '>
                      <div style='color:#999; font-size:12px; margin-bottom:8px;'>【英文】</div>
                      <div style='font-size:18px; line-height:1.6;'>{highlighted_english}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("詳細を見る", key="reveal_card", use_container_width=True):
                    st.session_state.card_revealed = True
                    st.rerun(scope="fragment")
                return

            # 裏面: 単語+IPA+類義語 + 英文 + 和訳 + 解説
            pronunciations = _parse_word_pronunciations(row["explanation"])
            synonyms = _parse_word_synonyms(row["explanation"])
            explanation_no_ipa = _strip_ipa(row["explanation"])

            words_block_html = ""
            for w in words_list:
                ipa = pronunciations.get(w.lower(), "")
                syns = synonyms.get(w.lower(), [])
                meta_parts = []
                if ipa:
                    meta_parts.append(
                        f"<span style='color:#4b5563;'>/{html.escape(ipa)}/</span>"
                    )
                if syns:
                    meta_parts.append(
                        f"<span style='color:#4b5563;'>≈ {html.escape(', '.join(syns))}</span>"
                    )
                meta_html = (
                    f"<div style='font-size:13px; color:#4b5563; margin-top:1px; "
                    f"display:flex; flex-wrap:wrap; gap:10px;'>{''.join(meta_parts)}</div>"
                    if meta_parts
                    else ""
                )
                words_block_html += (
                    f"<div style='margin-bottom:10px;'>"
                    f"<div style='font-weight:700; font-size:16px; color:#111827;'>{html.escape(w)}</div>"
                    f"{meta_html}"
                    f"</div>"
                )

            st.markdown(
                f"""
                <div style='
                    background: #fff; border: 1px solid #e5e7eb;
                    border-radius: 10px; padding: 14px 16px; margin: 8px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                '>
                  <div style='color:#999; font-size:11px; margin-bottom:2px;'>単語</div>
                  <div style='margin-bottom:10px;'>{words_block_html}</div>
                  <div style='color:#999; font-size:11px; margin-bottom:2px;'>英文</div>
                  <div style='font-size:17px; line-height:1.5; margin-bottom:10px;'>{highlighted_english}</div>
                  <div style='color:#999; font-size:11px; margin-bottom:2px;'>和訳</div>
                  <div style='font-size:14px; line-height:1.5; margin-bottom:10px;'>{html.escape(row["japanese"])}</div>
                  <div style='color:#999; font-size:11px; margin-bottom:2px;'>解説</div>
                  <div style='font-size:13px;'>{_format_explanation_html(explanation_no_ipa)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if words_list and os.getenv("UNSPLASH_ACCESS_KEY"):
                st.markdown(
                    "<div style='color:#999; font-size:11px; margin-top:8px; margin-bottom:4px;'>🖼️ 単語のイメージ</div>",
                    unsafe_allow_html=True,
                )
                selected_word = st.radio(
                    "単語",
                    words_list,
                    horizontal=True,
                    key=f"img_word_{row['id']}",
                    label_visibility="collapsed",
                )
                if selected_word:
                    images = get_or_fetch_images(row["id"], selected_word)
                    if not images:
                        st.caption("画像が見つかりませんでした。")
                    else:
                        cols = st.columns(len(images))
                        for c, img in zip(cols, images):
                            with c:
                                st.image(img["thumb"], use_container_width=True)
                                st.caption(f"📷 [{img['photographer']}]({img['photographer_url']})")

        _reveal_section()

        # ── わからない / わかる ボタンは表面・裏面ともに常時表示 ──
        is_last = idx == len(card_rows) - 1
        col_ng, col_ok = st.columns(2)
        with col_ng:
            if st.button("❌ わからない", key="mark_review", type="secondary", use_container_width=True):
                update_status(row["id"], "review")
                if is_last:
                    st.session_state.card_finished = True
                else:
                    st.session_state.card_index = idx + 1
                st.rerun()
        with col_ok:
            if st.button("✅ わかる", key="mark_mastered", type="primary", use_container_width=True):
                update_status(row["id"], "mastered")
                if is_last:
                    st.session_state.card_finished = True
                else:
                    st.session_state.card_index = idx + 1
                st.rerun()

        if st.session_state.get("card_finished"):
            st.success("🎉 このセットの最後のカードでした!")
            if st.button("最初から", key="restart_deck", use_container_width=True):
                st.session_state.card_index = 0
                st.session_state.card_finished = False
                st.session_state.pop("last_viewed_card_id", None)
                st.rerun()

        with st.expander("⚙️ このカードを削除"):
            if st.button("削除する", key=f"delete_card_{row['id']}", type="secondary"):
                delete_sentence(row["id"])
                st.session_state.pop("card_mode_rows", None)
                st.session_state.pop("card_index", None)
                st.rerun()
    else:
        # ── 単語リスト表示 ──
        st.caption("タップでカードを開く")
        for i, row in enumerate(rows):
            words_display = " / ".join(row["words"].split(","))
            status_icon = {"new": "🆕", "review": "🔁", "mastered": "✅"}.get(row.get("status", "new"), "🆕")
            view_n = row.get("view_count", 0)
            label = f"{status_icon}  {words_display}　({view_n}回)"
            if st.button(label, key=f"open_card_{row['id']}", use_container_width=True):
                st.session_state.card_mode_rows = rows
                st.session_state.card_index = i
                st.session_state.card_finished = False
                st.session_state.pop("last_viewed_card_id", None)
                st.rerun()
