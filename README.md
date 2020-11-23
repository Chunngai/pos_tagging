# pos_tagging

## Usage
### Virtual env
```bash
python3 -m venv .pos_tagging
source .pos_tagging/bin/activate
```

### Dependencies
```bash
pip3 install wheel sklearn torch transformers pytorch-crf
```

### Indonesian
1. Clean data.
2. Split the data into train, valid, test. The default valid and test ratio are both 0.1.
3. Train a model. Will output a dir [RESULT_DIR]. Specify the model with `--model`.
4. Test the model. Output of the test is saved in "[RESULT_DIR]/[CHECKPOINT_DIR]/results*.txt" and the accuracy in "[RESULT_DIR]/[CHECKPOINT_DIR]/accuracy\*.txt".

```bash
sh clean_ind.sh Ind_train.txt
python3 split.py --data-file Ind_train.txt.cleaned
python3 train.py --train Ind_train.txt.train --valid Ind_train.txt.valid --model cahya/bert-base-indonesian-1.5G
python3 test.py --test Ind_train.txt.test --checkpoint-dir [RESULT_DIR]/[CHECKPOINT_DIR]
```

### Lao
1. Clean data.
2. Split the data into train, valid, test. The default valid and test ratio are both 0.1.
3. Train a model. Will output a dir [RESULT_DIR]. Specify the model with `--model`.
4. Test the model. Output of the test is saved in "[RESULT_DIR]/[CHECKPOINT_DIR]/results*.txt" and the accuracy in "[RESULT_DIR]/[CHECKPOINT_DIR]/accuracy\*.txt".

```bash
sh clean_lao.sh Lao_train.tsv
python3 split.py --data-file Lao_train.tsv.cleaned
python3 train.py --train Lao_train.tsv.train --valid Lao_train.tsv.valid --model xlm-roberta-base
python3 test.py --test Lao_train.tsv.test --checkpoint-dir [RESULT_DIR]/[CHECKPOINT_DIR]
```

### N-fold splitting
Use `--folds` of split.py. For example:
```bash
python3 split.py --data-file Ind_train.txt.cleaned --folds 5
```
