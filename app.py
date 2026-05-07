#!/usr/bin/env python3
"""DUO 3.0風 ビジネス英語例文ジェネレーター - Streamlit Web App"""

import re

import anthropic
import streamlit as st
from dotenv import load_dotenv

from database import delete_sentence, get_all_sentences, init_db, save_sentence, search_sentences

load_dotenv()

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


# ── ページ設定 ────────────────────────────────────────────
st.set_page_config(page_title="DUO 3.0風 ビジネス英語ジェネレーター", page_icon="📚", layout="wide")
st.title("📚 DUO 3.0風 ビジネス英語例文ジェネレーター")

init_db()

tab_gen, tab_hist = st.tabs(["生成", "履歴"])

# ── タブ1: 生成 ───────────────────────────────────────────
with tab_gen:
    st.subheader("英単語を入力して例文を生成")
    words_input = st.text_area(
        "単語をスペース区切りで入力（3〜5語）",
        placeholder="例: negotiate deadline stakeholder",
        height=80,
    )

    if st.button("例文を生成", type="primary"):
        words = words_input.strip().split()
        if len(words) < 3 or len(words) > 5:
            st.error("3〜5語を入力してください。")
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
