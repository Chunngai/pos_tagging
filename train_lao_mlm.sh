#!/bin/bash

set -e
set -x

if [ ! -d .mlm ]
then
	python3 -m venv .mlm
fi
source .mlm/bin/activate

pip install wheel torch

if [ ! -d transformers ]
then
	git clone https://github.com/huggingface/transformers.git
fi
cd transformers
pip install -e . 

ROOT_DIR=lao_mlm
MODEL_DIR="$ROOT_DIR/model"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p $ROOT_DIR $MODEL_DIR $OUTPUT_DIR

cd $MODEL_DIR

wget -c https://huggingface.co/xlm-roberta-large/resolve/main/config.json
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/tokenizer.json

cd ..

python3 ../examples/language-modeling/run_mlm.py \
	--model_name_or_path $MODEL_DIR \
	--train_file $1 \
	--do_train \
	--output_dir $OUTPUT_DIR \
	--line_by_line


