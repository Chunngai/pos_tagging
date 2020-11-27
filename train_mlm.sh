#!/bin/bash

set -e
set -x

if [ $# -ne 4 ]; then exit; fi

DATA_PATH="$(cd $(dirname $2); pwd)/$2"

if [ ! -d .mlm ]; then python3 -m venv .mlm; fi
source .mlm/bin/activate
pip install wheel torch

if [ ! -d transformers ]; then git clone https://github.com/huggingface/transformers.git; fi
cd transformers
sed -i 's/^faiss-cpu$/faiss-cpu==1.6.3/' examples/requirements.txt
pip install -e . 
pip install -r examples/requirements.txt

ROOT_DIR="mlm/$1"
DATA_DIR="$ROOT_DIR/data"
MODEL_DIR="$ROOT_DIR/model"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p $ROOT_DIR $MODEL_DIR $OUTPUT_DIR $DATA_DIR

cp $DATA_PATH $DATA_DIR

# $2 should be a csv, a json or a txt file.
python3 examples/language-modeling/run_mlm.py \
	--model_name_or_path $1 \
	--train_file "$DATA_DIR/$2" \
	--do_train \
	--output_dir $OUTPUT_DIR \
	--line_by_line \
	--max_seq_length $3 \
	--num_train_epochs $4

