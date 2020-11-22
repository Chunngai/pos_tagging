# indonesian_pos_tagging

## Usage
1. Create a virtual env and activate it.
```bash
python3 -m venv .indonesian_pos_tagging
```

2. Install dependencies.
```bash
pip3 install wheel sklearn torch transformers pytorch-crf
```

3. Split the data into train, valid, test. The default valid and test ratio are both 0.1.
```bash
python3 split.py --data-file Ind_train.txt
```

4. Train a model. Will output a dir [RESULT_DIR]. Specify the model with `--model`.
```bash
python3 train.py --train Ind_train.train --valid Ind_train.valid --model bert-crf
```

5. Test the model. Output of the test is saved in "[RESULT_DIR]/[CHECKPOINT_DIR]/results*.txt" and the accuracy in "[RESULT_DIR]/[CHECKPOINT_DIR]/accuracy\*.txt".
```bash
python3 test.py --test Ind_train.test --checkpoint-dir ind_results/checkpoint-1500/
```
