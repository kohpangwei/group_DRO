import os
import sys
import numpy as np
import random
import string
import pandas as pd

################ Paths and other configs - Set these #################################

data_dir = '/u/scr/nlp/dro/multinli/data'
glue_dir = '/u/scr/nlp/dro/multinli/glue_data/MNLI'

type_of_split = 'random'
assert type_of_split in ['preset', 'random']
# If 'preset', use the official train/val/test MultiNLI split
# If 'random', randomly split 50%/20%/30% of the data to train/val/test

######################################################################################

def tokenize(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    s = s.split(' ')
    return s

### Read in data and assign train/val/test splits
train_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_train.jsonl'),
    lines=True)

val_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_dev_matched.jsonl'),
    lines=True)

test_df = pd.read_json(
    os.path.join(
        data_dir,
        'multinli_1.0_dev_mismatched.jsonl'),
    lines=True)

split_dict = {
    'train': 0,
    'val': 1,
    'test': 2
}

if type_of_split == 'preset':
    train_df['split'] = split_dict['train']
    val_df['split'] = split_dict['val']
    test_df['split'] = split_dict['test']
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

elif type_of_split == 'random':
    val_frac = 0.2
    test_frac = 0.3

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    n = len(df)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    n_train = n - n_val - n_test
    splits = np.array([split_dict['train']] * n_train + [split_dict['val']] * n_val + [split_dict['test']] * n_test)
    np.random.shuffle(splits)
    df['split'] = splits

### Assign labels
df = df.loc[df['gold_label'] != '-', :]
print(f'Total number of examples: {len(df)}')
for k, v in split_dict.items():
    print(k, np.mean(df['split'] == v))

label_dict = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}
for k, v in label_dict.items():
    idx = df.loc[:, 'gold_label'] == k
    df.loc[idx, 'gold_label'] = v

### Assign spurious attribute (negation words)
negation_words = ['nobody', 'no', 'never', 'nothing'] # Taken from https://arxiv.org/pdf/1803.02324.pdf

df['sentence2_has_negation'] = [False] * len(df)

for negation_word in negation_words:
    df['sentence2_has_negation'] |= [negation_word in tokenize(sentence) for sentence in df['sentence2']]

df['sentence2_has_negation'] = df['sentence2_has_negation'].astype(int)

## Write to disk
df = df[['gold_label', 'sentence2_has_negation', 'split']]
df.to_csv(os.path.join(data_dir, f'metadata_{type_of_split}.csv'))
