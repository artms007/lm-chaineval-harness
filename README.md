# lm-chaineval-harness
An chain version of lm evaluation framework.

このツールは、言語モデルの自動評価用フレームワークです。  
以下の機能を提供します。

- 既存の評価用データを用いた言語モデルの評価
- 既存の評価用データを活用した、逆翻訳による実行ベース評価での言語モデルの評価

## 環境設定

### Install

```shell
git clone https://github.com/KuramitsuLab/lm-chaineval-harness.git
cd lm-chaineval-harness
pip3 install -r requirements.txt
```

### API キー

評価に使用するモデルで、API キーが必要な場合は`.env` ファイルに記載して保存、もしくは環境変数に設定してください。

以下のAPIをサポートしています。
- Hugging Face のアクセストークン
- OpenAI API
- Amazon Bedrock API（boto3）

```plaintext:envファイル
OPENAI_API_KEY
HF_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

## 評価方法

1. [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) からテンプレートファイルを選ぶ、もしくは作成する
2. 任意のパス名に変更後、`chain.sh` として保存する
    ```sh
    python3 ./scripts/main.py \
        --model_path <MODEL_PATH> \
        --dataset <DATASET_PATH> \
        --template <TEMPLATE_PATH> \
        --metrics <METRIC_PATH> \
        --result_path <RESULT_PATH>
    ```
    
    - `model_path` : 評価したいモデルのパス名を指定
        - OpenAI モデルは先頭に `openai:` を付けて指定（e.g., `openai:gpt-4`）
        - Amazon Bedrock 経由モデルは先頭に `bedrock:` を付けて指定（e.g., `bedrock:anthropic.claude-v2:1`）
    - `dataset` : HuggingFace Hub 上で提供されているデータセットのパス名を指定
        - e.g., `openai_humaneval`, `kogi-jwu/jhumaneval`
        - 個人がローカルに所有するjsonl 形式のデータを指定することも可能
    - `template` : [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) から選んだテンプレートのパス名を指定
        - 個人で新たに作成したテンプレートのパス名の指定も可能
    - `metrics` : 評価指標のパス名を指定
        - [HuggingFaceのevaluate-metric](https://huggingface.co/evaluate-metric)で提供されている評価指標を使っています
        - 現在のサポート：`pass@1`
    - `result_path` : 結果を格納するファイル名を指定
        - 指定なしでも自動で結果のファイルを作成してくれます


3. 評価を実行する
    ```sh
    sh chain.sh
    ```

## 評価方法 - BackCodeEval

逆翻訳を活用した実行ベースでの評価方法をサポートしています。


1. [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) からテンプレートファイルを選ぶ、もしくは作成する
2. 任意のパス名に変更後、`chain.sh` として保存する
    ```sh
    # 評価したいタスク
    python3 ./scripts/main.py \
        --model_path <MODEL_PATH> \
        --dataset <DATASET_PATH> \
        --template <TEMPLATE_PATH_1> \
        --result_path <RESULT_PATH_1>
    
    # BackCodeEval
    python3 ./scripts/main.py \
        --model_path <MODEL_PATH> \
        --dataset <RESULT_PATH_1> \
        --template <TEMPLATE_PATH_2> \
        --metrics <METRIC_PATH> \
        --result_path <RESULT_PATH_2>
    ```
    
    - `model_path` : 評価したいモデルのパス名を指定
        - OpenAI モデルは先頭に `openai:` を付けて指定（e.g., `openai:gpt-4`）
        - Amazon Bedrock 経由モデルは先頭に `bedrock:` を付けて指定（e.g., `bedrock:anthropic.claude-v2:1`）
    - `dataset` : HuggingFace Hub 上で提供されているデータセットのパス名を指定
        - e.g., `openai_humaneval`, `kogi-jwu/jhumaneval`
        - 個人がローカルに所有するjsonl 形式のデータを指定することも可能
    - `template` : [`templates`](https://github.com/KuramitsuLab/lm-chaineval-harness/tree/main/templates) から選んだテンプレートのパス名を指定
        - 個人で新たに作成したテンプレートのパス名の指定も可能
    - `metrics` : 評価指標のパス名を指定
        - [HuggingFaceのevaluate-metric](https://huggingface.co/evaluate-metric)で提供されている評価指標を使っています
        - 現在のサポート：`pass@1`
    - `result_path` : 結果を格納するファイル名を指定
        - 指定なしでも自動で結果のファイルを作成してくれます


3. 評価を実行する
    ```sh
    sh chain.sh
    ```

## その他のオプション

### アクセストークンやAPI が必要なモデルの評価

モデルに合わせて必要なパラメータを追加してください。

```sh
# Hugging Face
--hf_token $HF_TOKEN

# OpenAI
--openai_api_key $OPENAI_API_KEY

# Amazon Bedrock
--aws_access_key_id $AWS_ACCESS_KEY_ID
--aws_secret_access_key $AWS_SECRET_ACCESS_KEY
```

### モデルのパラメータ設定

HuggingFace のPipeline で使用できるパラメータを個別で設定可能です。  

```sh
python3 ./scripts/main.py \
    --model_path <MODEL_PATH> \
    --max_new_tokens 512 \
    --temperature 0.1 \
    --top_p 0.90 \
    --dataset <DATASET_PATH> \
    --template <TEMPLATE_PATH> \
    --metrics <METRIC_PATH> \
    --result_path <RESULT_PATH>
```

### 量子化の有効化

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) を使用した4bitでの量子化を指定することができます。  
量子化を行う際には、コマンドライン引数として `use_4bit` を追加してください。

```sh
python3 ./scripts/main.py \
    --model_path <MODEL_PATH> \
    --dataset <DATASET_PATH> \
    --template <TEMPLATE_PATH> \
    --metrics <METRIC_PATH> \
    --result_path <RESULT_PATH> \
    --use_4bit
```

### テスト実行

`test_run` を追加すると、データセットの先頭5件だけを実行します。

```sh
python3 ./scripts/main.py \
    --model_path <MODEL_PATH> \
    --dataset <DATASET_PATH> \
    --template <TEMPLATE_PATH> \
    --metrics <METRIC_PATH> \
    --result_path <RESULT_PATH> \
    --test_run
```

ツールが動作するかを確認するには以下のような実行が有効です。

```sh
python3 ./scripts/main.py \
    --dataset openai_humaneval \
    --metrics pass@1 \
    --test_run
```