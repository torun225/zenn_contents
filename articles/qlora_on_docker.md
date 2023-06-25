---
title: "DockerでqLoraを行う方法"
emoji: "🧠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AI", "Docker", "LLM"]
published: false
---

最近 LLM にはまっているのですが、コンテナ上の jupyter で qLora を行っている記事が見られなかったためメモとして残します。
同じ環境で、Lora も可能なのを確認していますので、ミドルスペック以上の GPU が使える人であれば、colab の代わりに使えると思います。

# 検証した環境

| ハードウェア |              |
| ------------ | ------------ |
| メモリ       | 32GB         |
| GPU          | RTX3060 12GB |

| ソフトウェア   |           |
| -------------- | --------- |
| OS             | Windows10 |
| Docker Desktop | 4.20.1    |
| Docker         | 24.0.2    |
| Docker Compose | 2.18.1    |

# 環境構築

Docker を使って環境を構築します。

## GPU 環境の構築

前提として Docker で GPU を使える状態にする必要がありますので、下記の記事を参考に設定を行います。

https://qiita.com/nabion/items/4c4d4d4119c8586cbd9e
https://zenn.dev/takeguchi/articles/361e12a5321095

## Dockerfile と docker-compose.yml の設定

Dockerfile については、Nvidia が提供している CUDA 実行用の Ubuntu コンテナをベースにします。
password やディレクトリ、ネットワークの設定などは適宜変更してください。

```docker:Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

USER root

COPY ./requirements.txt /tmp
WORKDIR /root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git python3 python3-pip python3-distutils

RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='password'"]
CMD ["--notebook-dir=/root"]
```

requirements.txt には、コードの実行に必要なモジュールを記載します。
今回の検証では以下の内容としています。

```text:requirements.txt
jupyter
jupyterlab
tokenizers>=0.13.2
prompt_toolkit
numpy
autopep8
torch
torchvision
torchaudio
transformers==4.30.2
accelerate==0.20.3
sentencepiece
colorama
ctranslate2
git+https://github.com/lvwerra/trl.git
git+https://github.com/huggingface/peft.git
datasets
bitsandbytes
einops
wandb
scipy
protobuf==3.20.1
```

docker-compose.yml については以下の通りです。
deploy の項目で GPU を使えるようにしています。

```yml:docker-compose.yml
version: '3'
services:
  notebook:
    build: ./
    image: llm-notebook:1
    ports:
      - '8888:8888'
    volumes:
      - ./:/root
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [ '0' ]
              capabilities: [ gpu ]
```

# 実行

## コンテナの立ち上げ

上記で用意した 3 ファイル同じフォルダに保存したら、下記のコマンドでコンテナを起動します。

```sh
docker-compose up -d
```

起動を確認後にブラウザで「http://localhost:8888」にアクセスすることで、jupyter を開くことができます。
パスワードは Dockerfile で指定した「--NotebookApp.token」の値を入れてください。

## コードの実行

実行するコードは以下の記事を参考に作成しました。

https://note.com/__olender/n/ne9819f22b807
https://qiita.com/m__k/items/173ade78990b7d6a4be4
https://note.com/eurekachan/n/n899132477dff
https://note.com/npaka/n/nc387b639e50e
https://zenn.dev/syoyo/articles/6918defde8a807

正直、機械学習は詳しくないので、以下のコードやパラメータが適切かは謎ですが、cyberagent/open-calm-7b のモデルを kunishou/databricks-dolly-15k-ja のデータセットで学習して、会話できる状態にもっていきます。
実際に検証した際は eval_steps と logging_steps を 50 に設定して、RTX3060 12GB で 24 時間程度掛かりました。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

# 基本パラメータ
base_model = "cyberagent/open-calm-7b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "qlora-calm-7b-result"
output_dir = "qlora-calm-7b-output"

# トレーニング用パラメータ
eval_steps = 100
save_steps = 200
logging_steps = 100
epochs = 3
max_steps = 0 # ステップを指定すればepochsを無視して、max_stepsで終了する

# LoRA用パラメータ
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

# 他パラメータ
VAL_SET_SIZE = 0.1 # 検証分割比率
CUTOFF_LEN = 512  # コンテキスト長の上限

# ベースモデル量子化パラ設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ベースモデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# PEFT(LoRA)の設定
config = LoraConfig(r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    inference_mode=False,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=[
                            "query_key_value",
                            "dense",
                            "dense_h_to_4h",
                            "dense_4h_to_h",
                        ]
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # 学習可能パラメータの確認

# トークナイズ関数
def tokenize(prompt, tokenizer):
    result = tokenizer(prompt,
                       truncation=True,
                       max_length=CUTOFF_LEN,
                       padding=False,
    )
    return {"input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
    }

# データセットの準備
data = load_dataset(dataset)

# プロンプトテンプレートの準備
eos_token = tokenizer.decode([tokenizer.eos_token_id])
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""ユーザー:{data_point["instruction"]}
入力:{data_point["input"]}
システム:{data_point["output"]}{eos_token}"""
    else:
        result = f"""ユーザー:{data_point["instruction"]}
システム:{data_point["output"]}{eos_token}"""

    return result

# 学習データと検証データの準備
train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

# トレーナーの定義
trainer = Trainer(
    model = model,
    train_dataset = train_data,
    eval_dataset = val_data,
    args = TrainingArguments(
        num_train_epochs=epochs,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",
        save_total_limit=10,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator= DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()

trainer.model.save_pretrained(peft_name)    # LoRAモデルの保存
tokenizer.save_pretrained(peft_name)
```

以下のコードで学習したモデルをロードします。
実行する前に VRAM を解放するために、python のカーネルを再起動してください。

```python

import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 基本パラメータ
base_model = "cyberagent/open-calm-7b"
peft_name = "qlora-calm-7b-result"

# 入力文章
input_text = "カレーにジャガイモは入れるべき？"

# ベースモデル量子化パラ設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ベースモデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map='auto',
)
# Rinnaのトークナイザーでは、「use_fast=False」も必要になる
tokenizer = AutoTokenizer.from_pretrained(base_model)

# PEFT(LoRA)の読み込み
model = PeftModel.from_pretrained(model, peft_name)
model.eval()# 評価モード

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    result = f"""ユーザー:{data_point["instruction"]}
システム:"""

    return result

# テキスト生成関数の定義
def textgen(instruction,input=None,maxTokens=512):
    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=maxTokens,
                             do_sample=True,
                             temperature=0.7,
                             top_p=0.75,
                             top_k=40,
                             no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()
    #print(tokenizer.decode(outputs))

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "システム:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc+len(sentinel):]
            return result
        else:
            return('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        return('Warning: no <eos> detected ignoring output')

print("end")

```

ロードが完了したら下記のコードで推論を行います。

```python
instruction_text = """夏の飲み物と言えば何がある？"""
input_text = """"""

start_time = time.time()

print(f"ユーザー: {instruction_text}\nシステム: {textgen(instruction=instruction_text,input=input_text)}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
```

以下のような出力結果が得られます。
そこそこ、会話できているように見えます。

```text
ユーザー: 夏の飲み物と言えば何がある？
システム: 夏の定番ドリンクといえば、レモネード、アイスクリーム、コーラ、ビール、トマトジュース、アイスキューブ、アイスコーヒー、そして水です。
Elapsed time: 12.459978818893433 seconds
```

# まとめ

Docker 上で jupyter を用いて、qLora ができることを確認しました。
コンテナにすることでバージョン問題が発生した場合などに安全に対応できますので、参考になりましたら幸いです。

#### 宣伝

宣伝になりますが、AI を用いたキャラクターの作成をしてたりします。
今のところ chatGPT を用いていますが、近いうちに上記の方法で学習させたローカル LLM で動かしたいなと思っていますので、チャンネル登録やフォローして頂けると幸いです！

https://www.youtube.com/channel/UCZvXiHxvaCG7PcmRGpnP-WQ
https://twitter.com/MitsueMiko_ai
