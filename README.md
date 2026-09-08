# AGPS-Pyscore

## 概要
本システムは、API経由でリクエストされた申込レコードに対してスコアを予測し、予測結果をレスポンスします。\
詳細は要件定義書を見てください。

## 主要技術

| カテゴリ          | 技術・ツール               | 概要                                             |
| ---------------- | ------------------------- | ------------------------------------------------ |
| 開発言語          | Python3.11                | 豊富な機械学習ライブラリとFastAPIとの親和性         |
| APIフレームワーク | FastAPI                    | 高速な非同期処理と自動ドキュメント作成              |
| 推論エンジン      | ONNX Runtime               | 推論時間の高速化とモデルの書き換え防止              |
| サーバー管理      | Gunicorn + Uvicorn         | 本番環境でのプロセス管理とマルチワーカー実行の安定化 |
| 環境構築          | venv + pip(pyproject.toml) | サーバー環境の汚染防止とライブラリのコンフリクト防止 | 

## API仕様書

1. `uvicorn AGPS_pyscore.main:app`でAPIサーバーを起動
2. `http://127.0.0.1:8000/docs`にアクセス

## 環境構築

GunicornはWindows搭載のSAS端末で動作しなかったです。すみません、、、

1. 指定したバージョンのPythonを環境にインストール
2. `agps_pyscore-x.x.tar.gz`を解凍し、子ディレクトリを親ディレクトリに移動
3. ～は下記bashコマンド参照

```bash
> cd agps_pyscore_x.x  # ディレクトリに移動
> python -m venv .venv  # 仮想環境作成
> . .venv/bin/activate  # 仮想環境をアクティブ化する
(.venv) > pip install .  # Pythonライブラリのインストール
(.venv) > pip install --no-deps -r requirements.txt  # 上記で動かない場合はこっちでインストール
(.venv) > uvicorn AGPS_pyscore.main:app # APIサーバー起動コマンド実行
```

### パラメータ設定

`settings.py`でサーバー環境や希望レスポンス時間に合わせて設定してください。

- `intra_op_um_threads`: スコアリングモデル推論時のスレッド数
- `timeout`: リクエストからレスポンスまでのタイムアウト秒数

```Python
# settings.py

class PyscoreSettings(BaseSettings):
    intra_op_num_threads: int = 2
    timeout: float = 300 / 1000
```

## ディレクトリ構成

下記は`agps_pyscore_x.x.tar.gz`解凍後の構成です。

```
AGPS_pyscore
├── AGPS_pyscore
│   ├── main.py
│   ├── schema.py
│   ├── settings.py - 設定ファイル
│   ├── router
│   │   ├── new_pyscore.py
│   │   └── repeater_pyscore.py
│   ├── scoring
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   └── preprocessing.py
│   └── static
├── models
│   ├── new_model_vn.onnx - 第n次新規スコアのモデル
│   └── repeater_model_vn.onnx - 第n次新規スコアのモデル
├── .gitignore
├── PKG_INFO
├── pyproject.toml
├── README.md
└── requirements.txt
```


## テストデータ一覧


### 新規

新規スコア導入の目途が立ってから追記

### 途上


## パッケージ化

memo memo

```bash
# `pyproject.toml`の`version`を修正してから実行
python -m build --no-isolation --sdist
```