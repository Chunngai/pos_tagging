# indonesian_pos_tagging

## Usage
1. Create a virtual env and activate it.
```bash
python3 -m venv .indonesian_pos_tagging
```

2. Install dependencies.
```bash
pip3 install wheel sklearn torch transformers
```

3. Split the data into train, valid, test. The default valid and test ratio are both 0.1. Names of the output files are in the form of [FILE_BASE_NAME].train, [FILE_BASE_NAME].valid, [FILE_BASE_NAME].test, such as Ind_train.train, Ind_train.valid, Ind_train.test.
```bash
python3 split.py --data-file Ind_train.txt
```

4. Train the model. Will output a dir with a name in the form of "ind_results_[mm]\_[dd]\_[HH]_[MM]" such as ind_results_11_10_17_00
```bash
python3 train.py --train Ind_train.train --valid Ind_train.valid
```

5. Test the model. The checkpoint dir is from the model dir such as "ind_result_11_10_17_00/checkpoint-1500". Output of the test is saved in "ind_result_11_10_17_00/checkpoint-1500/results.txt" and the accuracy in "ind_result_11_10_17_00/checkpoint-1500/accuracy.txt".
```bash
python3 test.py --test Ind_train.test --checkpoint-dir ind_results/checkpoint-1500/
```
