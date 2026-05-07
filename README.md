# DUO 3.0風 ビジネス英語例文ジェネレーター

DUO 3.0のスタイルにインスパイアされた、Anthropic APIを使ったビジネス英語例文生成CLIツールです。  
3〜5個の英単語を渡すと、TOEIC800点以上レベルの自然な例文・和訳・解説を生成します。

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. APIキーの設定

`.env.example` をコピーして `.env` を作成し、APIキーを設定します。

```bash
cp .env.example .env
```

`.env` を編集：

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

APIキーは [Anthropic Console](https://console.anthropic.com/) から取得できます。

## Webアプリ（Streamlit）

### 起動方法

```bash
streamlit run app.py
```

ブラウザが自動で開き `http://localhost:8501` でアクセスできます。

### 機能

| タブ | 機能 |
|---|---|
| 生成 | 単語をスペース区切りで入力して例文を生成。結果を英文・和訳・解説で表示し、自動でDBに保存。 |
| 履歴 | 過去の生成結果を新しい順に一覧表示。クリックで全文展開、削除ボタン付き。キーワード検索も可能。 |

---

## CLIの使い方

```bash
python main.py <単語1> <単語2> <単語3> [<単語4>] [<単語5>]
```

### 例

```bash
python main.py negotiate deadline stakeholder
```

**出力例：**

```
【英文】
We need to negotiate with key stakeholders before the deadline to ensure all parties are aligned on the project scope.

【和訳】
プロジェクトの範囲についてすべての関係者が合意できるよう、締め切り前に主要なステークホルダーと交渉する必要があります。

【解説】
- negotiate（交渉する）: ビジネスでは合意形成のプロセスを指す。"negotiate a deal/contract/terms" のように使う。
- deadline（締め切り）: プロジェクト管理の基本語。"meet/miss a deadline"（期限を守る/守れない）の形が頻出。
- stakeholder（利害関係者）: 意思決定に影響を受けるすべての関係者。経営層から顧客まで幅広く使われるビジネス必須語。
```

---

```bash
python main.py leverage synergy prioritize streamline
```

### オプション

| オプション | 説明 |
|---|---|
| `--verbose` | トークン使用量とキャッシュヒット状況を表示 |

```bash
python main.py negotiate deadline stakeholder --verbose
```

## 仕様

| 項目 | 内容 |
|---|---|
| 使用モデル | `claude-sonnet-4-5` |
| Temperature | 0.7（表現の多様性を確保） |
| 入力単語数 | 3〜5個 |
| プロンプトキャッシュ | システムプロンプトにキャッシュを適用 |
| 出力形式 | 英文 / 和訳 / 解説 の3ブロック |

## ファイル構成

```
duosystem/
├── app.py            # Streamlit Webアプリ
├── database.py       # SQLite操作モジュール
├── history.db        # 生成履歴DB（自動生成）
├── main.py           # CLI スクリプト
├── requirements.txt  # 依存パッケージ
├── .env.example      # APIキー設定テンプレート
├── .env              # APIキー（gitignore対象）
└── README.md         # このファイル
```

## 注意事項

- `.env` ファイルはバージョン管理に含めないでください（`.gitignore` に追加推奨）
- APIの利用にはAnthropicのAPIキーと利用料金が必要です
