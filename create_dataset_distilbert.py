# This script is part of a larger research project shown at https://github.com/mmeberg/PyVulDet-NER

import pandas as pd
import os
import requests
import time
import sys
import json
from requests_oauthlib import OAuth1Session
from requests_oauthlib import OAuth1
import base64
from collections import Counter, defaultdict
import transformers
import random
import datasets
import tokenize
import io
import re
import time
import math
import datetime
import pickle
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split


def create_labels(row):
    tags = row['ner_tags']
    labels = []
    to_change = tags
    for x in to_change:
        x[0] = -100
        x[-1] = -100
        pad = 512 - len(x)
        x += [0]*pad
        labels.append(x)
    return {'labels':labels}

def change_attention(row):
    attention = row['attention_mask']
    atts = []
    to_change = attention
    for x in to_change:
        if 0 in x:
            indx_to_change = x.index(0)
            x[indx_to_change-1] = 0
        else:
            x[-1] = 0
        x[0] = 0
        atts.append(x)
    return {'attention_mask':atts}

def DistilBERT_tokenizer_tag(row):
    mode = row.cwetype
    content = row.short_text

    tok_results = []

    token_dict = tokenizer(content.strip(), truncation=True, max_length = 512,  padding='max_length')
    input_ids = token_dict['input_ids']
    attention_mask = token_dict['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(token_dict.input_ids)

    token_list = [tok for tok in tokens if tok != '[SEP]' and tok != '[PAD]' and tok != '[CLS]']

    if row.type == 'bp':
        bp = row.parts

        bp_token_dict = tokenizer(bp.strip(), max_length = 512, truncation = True)
        bp_tokens = tokenizer.convert_ids_to_tokens(bp_token_dict.input_ids)
        bp_token_list = [bp_tok for bp_tok in bp_tokens if bp_tok != '[SEP]' and bp_tok != '[PAD]' and bp_tok != '[CLS]']

        if bp_token_list == [] or len(set(bp_token_list)) == 1:
            return tok_results
        else:
            if bp_token_list != [] and len(bp_token_list) > 1:
                while bp_token_list[-1] == '':
                    bp_token_list.pop()

        tester = []
        for i in range(len(token_list)-len(bp_token_list)):
            # Not looking at the first and last because that was causing mis-match issues
            if token_list[i:i+len(bp_token_list)] == bp_token_list:
                tester.append(i)
                tester.append(i+len(bp_token_list))

        if tester == []:

            tester3 = []
            for i in range(len(token_list)-len(bp_token_list)):
                if token_list[i+1:i+len(bp_token_list)+1] == bp_token_list:
                    tester3.append(i)
                    tester3.append(i+len(bp_token_list))
            if tester3 == []:
                return tok_results
            else:
                ner_tags = tag_vulparts(token_list, mode, tester3)
        else:
            ner_tags = tag_vulparts(token_list, mode, tester)

        tok_results.append({"ner_tags": ner_tags, 'tokens': token_list,
                            'input_ids':input_ids, 'attention_mask':attention_mask})

    else:
        ner_tags = [0]*len(token_list)
        tok_results.append({"ner_tags": ner_tags, 'tokens': token_list,
                            'input_ids':input_ids, 'attention_mask':attention_mask})

    return tok_results


#~~~~~~~~~~~~~~~~~~~

clean_and_short_file = sys.argv[0]
tokenizer_type = sys.argv[1]
    
#opening file
with open(clean_and_short_file, 'rb') as input:
    df = pickle.load(input)

if tokenizer_type == 'distilbert' or tokenizer_type == 'DistilBERT':
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
else:
    print('error with input')
    break

%%time
df = df[~df.short_text.str.startswith('cdef')]
df = df[~df.parts.str.strip().str.startswith('#')]
df = df.drop_duplicates()

df_bp = df.copy()
df_bp = df_bp[df_bp.type == 'bp']
df_bp.reset_index(inplace=True, drop=True)

%%time
tag_tok_results_bp = []
for x in range(0, len(df_bp), 25000):
    df_test = df_bp.copy()
    df_test = df_test.iloc[x: x+25000]
    tag_tok_results_bp.append(df_test.apply(DistilBERT_tokenizer_tag, axis =1))
    print(len(df_bp)-x)
    
tag_tok_results_bp_all = [y for x in tag_tok_results_bp for y in x]
bp_ner_tags = []
bp_tokens = []
bp_attention = []
bp_input_ids = []
for x in tag_tok_results_bp_all:
    if len(x) == 0:
    bp_ner_tags.append('None')
    bp_tokens.append('None')
    bp_attention.append('None')
    bp_input_ids.append('None')
    else:
        for y in x:
            bp_ner_tags.append(y['ner_tags'])
            bp_tokens.append(y['tokens'])
            bp_attention.append(y['attention_mask'])
            bp_input_ids.append(y['input_ids'])

df_bp['ner_tags'] = bp_ner_tags
df_bp['tokens'] = bp_tokens
df_bp['attention_mask'] = bp_attention
df_bp['input_ids'] = bp_input_ids
df_bp = df_bp.drop(columns = ['bp_len'])
df_bp = df_bp[df_bp.ner_tags != 'None']
df_bp.reset_index(drop=True, inplace=True)

#saving just in case
import pickle
with open('dataset_short_tag_tok_bp_ds.pickle', 'wb') as output:
    pickle.dump(df_bp, output)

df_bp = df_bp.drop(columns = ['text'])
df_bp = df_bp[['cwetype', 'type', 'short_text', 'parts', 'ner_tags', 'tokens', 'attention_mask','input_ids']]
df_bp['len_set_parts'] = df_bp['parts'].apply(lambda x: len(set(x)))
df_bp_50 = df_bp.copy()
df_bp_50 = df_bp_50[df_bp_50.parts.str.len() >= 50]
df_bp_50['parts_stripped'] = df_bp_50['parts'].apply(lambda x: x.strip())
df_bp_50['len_set_parts'] = df_bp_50['parts_stripped'].apply(lambda x: len(set(x)))
df_bp_50 = df_bp_50[df_bp_50.len_set_parts >= 5]
df_final = df_bp_50.copy()

#saving just in case
import pickle
with open('df_tagsandtokens_ds.pickle', 'wb') as output:
    pickle.dump(df_final, output)
    
df_final['ner_len'] = [len(x) for x in df_final.ner_tags]
df_final = df_final[df_final.short_text != '']
df_final = df_final.drop(columns = ['ner_len','len_set_parts', 'parts_stripped'])
print(df_final.cwetype.value_counts())

%%time
train, test = train_test_split(df_final, test_size=0.4, random_state=2023, shuffle = True, stratify = df_final.cwetype)

print('train counts:\n', train.cwetype.value_counts())
train_data = datasets.Dataset.from_pandas(train)
print('len of train data', len(train_data))
print()

valid, test2 = train_test_split(test, test_size=0.5, random_state=2023, shuffle = True, stratify = test.cwetype)
print('test counts:\n',test2.cwetype.value_counts())
test_data = datasets.Dataset.from_pandas(test2)
print('len of test data', len(test_data))
print()

print('valid counts:\n', valid.cwetype.value_counts())
valid_data = datasets.Dataset.from_pandas(valid)
print('len of valid data', len(valid_data))
print()

dataset = datasets.DatasetDict({"train":train_data,'validation':valid_data, 'test':test_data})
dataset_labels = dataset.map(create_labels, batched=True)
dataset_v2 = dataset_labels.map(change_attention, batched=True)

print('saving model ready data')
with open('distilbert_tagsandtokens.pickle', 'wb') as output:
    pickle.dump(dataset_v2, output)