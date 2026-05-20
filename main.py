#!/usr/bin/env python3
"""単語ジェネ - ビジネス英単語から例文を生成するCLIツール"""

import sys
import anthropic
from dotenv import load_dotenv

load_dotenv()

# システムプロンプトは静的なのでキャッシュ対象
SYSTEM_PROMPT = """あなたはビジネス英語の熟練講師です。TOEIC800点以上レベルの自然なビジネス英語例文を作成します。

以下のガイドラインに従ってください：
- 実際のビジネスシーンで使われる自然な英語を使用する
- 指定された単語をすべて文法的に自然な形で組み込む
- 例文はメール、会議、プレゼンテーション等のビジネス場面を想定する
- 和訳は自然な日本語ビジネス表現にする
- 解説は各単語のコアな意味とビジネス文脈での使い方を簡潔に説明する"""


def generate_sentence(words: list[str]) -> None:
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
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)

    # キャッシュ状況をデバッグ出力（--verboseオプション時）
    if "--verbose" in sys.argv:
        usage = response.usage
        print(f"\n[Token usage]")
        print(f"  Input:          {usage.input_tokens}")
        print(f"  Cache write:    {usage.cache_creation_input_tokens or 0}")
        print(f"  Cache read:     {usage.cache_read_input_tokens or 0}")
        print(f"  Output:         {usage.output_tokens}")


def main() -> None:
    # --verboseフラグを除いた引数を取得
    args = [a for a in sys.argv[1:] if a != "--verbose"]

    if len(args) < 3 or len(args) > 5:
        print("使い方: python main.py <単語1> <単語2> <単語3> [<単語4>] [<単語5>]")
        print()
        print("例:")
        print("  python main.py negotiate deadline stakeholder")
        print("  python main.py leverage synergy prioritize streamline")
        print()
        print("オプション:")
        print("  --verbose   トークン使用量を表示")
        sys.exit(1)

    generate_sentence(args)


if __name__ == "__main__":
    main()
