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

pip install -r examples/requirements.txt

ROOT_DIR=ind_mlm
DATA_DIR="$ROOT_DIR/data"
MODEL_DIR="$ROOT_DIR/model"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p $ROOT_DIR $MODEL_DIR $OUTPUT_DIR $DATA_DIR

cp "../$1" $DATA_DIR

cd $MODEL_DIR

wget -c https://huggingface.co/cahya/bert-base-indonesian-1.5G/resolve/main/config.json
wget -c https://huggingface.co/cahya/bert-base-indonesian-1.5G/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/cahya/bert-base-indonesian-1.5G/resolve/main/special_tokens_map.json
wget -c https://huggingface.co/cahya/bert-base-indonesian-1.5G/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/cahya/bert-base-indonesian-1.5G/resolve/main/vocab.txt

cd ../..

# $1 should be a csv, a json or a txt file.
python3 examples/language-modeling/run_mlm.py \
	--model_name_or_path $MODEL_DIR \
	--train_file "$DATA_DIR/$1" \
	--do_train \
	--output_dir $OUTPUT_DIR \
	--line_by_line


