#!/usr/bin/env python3
"""DUO 3.0風 ビジネス英語例文ジェネレーター - Streamlit Web App"""

import base64
import hashlib
import os
import re

import anthropic
import streamlit as st
from dotenv import load_dotenv

from database import delete_sentence, get_all_sentences, init_db, save_sentence, search_sentences

load_dotenv()

if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

SYSTEM_PROMPT = """あなたはビジネス英語の熟練講師です。TOEIC800点以上レベルの自然なビジネス英語例文を作成します。

以下のガイドラインに従ってください：
- 実際のビジネスシーンで使われる自然な英語を使用する
- 指定された単語をすべて文法的に自然な形で組み込む
- 例文はメール、会議、プレゼンテーション等のビジネス場面を想定する
- 和訳は自然な日本語ビジネス表現にする
- 解説は各単語のコアな意味とビジネス文脈での使い方を簡潔に説明する"""


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
（各単語の意味とビジネス文脈での使い方）"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0.7,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    return _parse_response(text)


EXTRACTION_PROMPT = """この画像は英語学習アプリ「abceed」のスクリーンショットです。
画像内で「色付き(オレンジ・赤・黄色など)で強調表示されている英単語」を抽出してください。

abceedでは、覚えるべき重要語が文章中でオレンジや赤などのアクセントカラーで強調表示されます。
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


# ── ページ設定 ────────────────────────────────────────────
st.set_page_config(page_title="DUO 3.0風 ビジネス英語ジェネレーター", page_icon="📚", layout="wide")
st.title("📚 DUO 3.0風 ビジネス英語例文ジェネレーター")

init_db()

tab_gen, tab_hist = st.tabs(["生成", "履歴"])

# ── タブ1: 生成 ───────────────────────────────────────────
with tab_gen:
    st.subheader("英単語を入力して例文を生成")

    with st.expander("📷 画像から苦手単語を抽出(abceedスクショ対応)"):
        uploaded = st.file_uploader(
            "abceedの苦手単語リストなどのスクショをアップロード",
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

                    save_sentence(words, result["english"], result["japanese"], result["explanation"])
                    st.caption("💾 履歴に保存しました")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# ── タブ2: 履歴 ───────────────────────────────────────────
with tab_hist:
    st.subheader("生成履歴")

    search_query = st.text_input(
        "キーワード検索（英文・和訳・単語から横断）",
        placeholder="例: negotiate",
    )

    rows = search_sentences(search_query) if search_query else get_all_sentences()

    if not rows:
        st.info("履歴がありません。" if not search_query else "検索結果が見つかりませんでした。")
    else:
        st.caption(f"{len(rows)} 件")
        for row in rows:
            words_display = " / ".join(row["words"].split(","))
            preview = row["english"][:60] + "…" if len(row["english"]) > 60 else row["english"]
            label = f"🕐 {row['created_at']}　|　{words_display}　|　{preview}"

            with st.expander(label):
                st.markdown("**【英文】**")
                st.write(row["english"])
                st.markdown("**【和訳】**")
                st.write(row["japanese"])
                st.markdown("**【解説】**")
                st.write(row["explanation"])

                if st.button("削除", key=f"delete_{row['id']}", type="secondary"):
                    delete_sentence(row["id"])
                    st.rerun()
