#!/bin/bash

set -e
set -x

DATA_PATH="$(cd $(dirname $1); pwd)/$1"

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

pip install -r examples/requirements.txt

ROOT_DIR=lao_mlm
DATA_DIR="$ROOT_DIR/data"
MODEL_DIR="$ROOT_DIR/model"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p $ROOT_DIR $MODEL_DIR $OUTPUT_DIR $DATA_DIR

cp $DATA_PATH $DATA_DIR

cd $MODEL_DIR

wget -c https://huggingface.co/xlm-roberta-large/resolve/main/config.json
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model
wget -c https://huggingface.co/xlm-roberta-large/resolve/main/tokenizer.json

cd ../..

# $1 should be a csv, a json or a txt file.
python3 examples/language-modeling/run_mlm.py \
	--model_name_or_path $MODEL_DIR \
	--train_file "$DATA_DIR/$1" \
	--do_train \
	--output_dir $OUTPUT_DIR \
	--line_by_line \
	# TODO
	--max_seq_length 128


