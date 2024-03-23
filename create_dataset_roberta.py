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


def CODEBERT_tokenizer_tag_v2(row):
    mode = row.cwetype
    content = row.short_text

    new_info = []
    if 'cdef ' in content[:50]:
        return []
    if len(content) > 100000:
        return []

    token_dict = tokenizer(content.strip(), truncation=False)
    input_ids = token_dict['input_ids']
    attention_mask = token_dict['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(token_dict.input_ids)
    token_list = [tok for tok in tokens if tok != '<s>' and tok != '<pad>' and tok != '</s>']
    tok_results = []

    if row.type == 'bp':
        bp = row.parts

        bp_token_dict = tokenizer(bp.strip(), truncation = False)
        bp_tokens = tokenizer.convert_ids_to_tokens(bp_token_dict.input_ids)
        bp_token_list = [bp_tok for bp_tok in bp_tokens if bp_tok != '<s>' and bp_tok != '<pad>' and bp_tok != '</s>']

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

            bp_token_list[0] = 'Ġ'+bp_token_list[0]
            tester2 = []
            for i in range(len(token_list)-len(bp_token_list)):
                if token_list[i:i+len(bp_token_list)] == bp_token_list:
                    tester2.append(i)
                    tester2.append(i+len(bp_token_list))
            if tester2 == []:
                tester3 = []
                for i in range(len(token_list)-len(bp_token_list)):
                    if token_list[i+1:i+len(bp_token_list)] == bp_token_list[1:]:
                        tester3.append(i)
                        tester3.append(i+len(bp_token_list))
                if tester3 == []:
                    return tok_results
                else:
                    ner_tags = tag_vulparts_v2(token_list, mode, tester3, token_dict)
            else:
                ner_tags = tag_vulparts_v2(token_list, mode, tester2, token_dict)
        else:
            ner_tags = tag_vulparts_v2(token_list, mode, tester, token_dict)

        token_list.append('</s>')
        token_list.insert(0, '<s>')
        ner_tags.append(-100)
        ner_tags.insert(0, -100)
        tok_results.append({"ner_tags": ner_tags, 'tokens': token_list,
                            'input_ids':input_ids, 'attention_mask':attention_mask})

    else:
        token_list.append('</s>')
        token_list.insert(0, '<s>')
        ner_tags = [0]*len(token_list)
        tok_results.append({"ner_tags": ner_tags, 'tokens': token_list,
                            'input_ids':input_ids, 'attention_mask':attention_mask})

    return tok_results

def shorten_data_with_windows_after_tokenization(infodict):
    '''
    Shortening after tokenization
    '''
    new_info = []
    new_size = 510

    tags = infodict['ner_tags']
    tokens = infodict['tokens']
    attn = infodict['attention_mask']
    input_ids = infodict['input_ids']
    tag_indeces = [i for i, x in enumerate(tags) if x != 0 and x != -100]

    # If there are more than 512 tokens
    if len(tokens) >= new_size:
        # If the indeces start and stop before 512, cut the list to 512
        if tag_indeces[0] < new_size and tag_indeces[-1] <= new_size:
            breaks = [i for i, x in enumerate(tokens) if '\n' in x or ')' in x or ']' in x or '}' in x or '\t' in x]
            if [m for m in breaks if m <= new_size] == []:
                new_info.append({'new_tokens': [], 'new_tags': [],
                            'new_attn': [], 'new_input_ids': []})
                return new_info
            else:
                break_id_to_use = [m for m in breaks if m <= new_size][-1]
                new_tokens = tokens[:break_id_to_use]
                new_tags = tags[:break_id_to_use]
                new_attn = attn[:break_id_to_use]
                new_input_ids = input_ids[:break_id_to_use]
                new_info.append({'new_tokens': new_tokens, 'new_tags': new_tags,
                                'new_attn': new_attn, 'new_input_ids': new_input_ids})

        elif len(tag_indeces) <= new_size:
            breaks = [i for i, x in enumerate(tokens) if '\n' in x or ')' in x or ']' in x or '}' in x]
            if [m for m in breaks if m <= tag_indeces[0]] == []:
                new_info.append({'new_tokens': [], 'new_tags': [],
                            'new_attn': [], 'new_input_ids': []})
                return new_info
            else:
                break_id_to_use1 = [m for m in breaks if m <= tag_indeces[0]][-1]
                break_id_to_use2 = [m for m in breaks if m >= break_id_to_use1 and m <= break_id_to_use1+new_size][-1]
                new_tokens = tokens[break_id_to_use1:break_id_to_use2]
                new_tags = tags[break_id_to_use1:break_id_to_use2]
                new_attn = attn[break_id_to_use1:break_id_to_use2]
                new_input_ids = input_ids[break_id_to_use1:break_id_to_use2]
                new_info.append({'new_tokens': new_tokens, 'new_tags':new_tags,
                                'new_attn': new_attn, 'new_input_ids': new_input_ids})
        else:
            breaks = [i for i, x in enumerate(tokens) if '\n' in x or ')' in x or ']' in x or '}' in x]
            break_ids_to_use = []
            for i in range(math.ceil(len(tokens)/new_size)):
                if i == 0:
                    if [m for m in breaks if m <=new_size] == []:
                        break_id = breaks[0]
                    else:
                        break_id = [m for m in breaks if m <=new_size][-1]
                    chunk_start = 0
                else:
                    break_id = [m for m in breaks if m >= break_ids_to_use[i-1] and m <= break_ids_to_use[i-1]+new_size][-1]
                    chunk_start = break_ids_to_use[i-1]
                break_ids_to_use.append(break_id)
                chunk = [chunk_start, break_id]
                new_tokens = tokens[chunk[0]:chunk[1]]
                new_tags = tags[chunk[0]:chunk[1]]
                new_attn = attn[chunk[0]:chunk[1]]
                new_input_ids = input_ids[chunk[0]:chunk[1]]
                new_info.append({'new_tokens': new_tokens, 'new_tags':new_tags,
                                'new_attn': new_attn, 'new_input_ids': new_input_ids})

    else:
        ## If there are not more than 512 tokens
        new_tokens = tokens
        new_tags = tags
        new_attn = attn
        new_input_ids = input_ids
        new_info.append({'new_tokens': new_tokens, 'new_tags':new_tags,
                        'new_attn': new_attn, 'new_input_ids': new_input_ids})

    return new_info

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

#~~~~~~~~~~~~~~~~~~~~~

clean_and_short_file = sys.argv[0]
tokenizer_type = sys.argv[1]
    
#opening file
with open(clean_and_short_file, 'rb') as input:
    df = pickle.load(input)

if tokenizer_type == 'codebert' or tokenizer_type == 'CodeBERT':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
elif tokenizer_type == 'roberta' or tokenizer_type == 'RoBERTa':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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
for x in range(0, len(df_bp), 50000):
    df_test = df_bp.copy()
    df_test = df_test.iloc[x: x+50000]
    tag_tok_results_bp.append(df_test.apply(CODEBERT_tokenizer_tag_v2, axis =1))
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
with open('dataset_short_tag_tok_bp.pickle', 'wb') as output:
    pickle.dump(df_bp, output)
    
df_bp = df_bp.drop(columns = ['text'])
df_bp = df_bp[['cwetype', 'type', 'short_text', 'parts', 'ner_tags', 'tokens', 'attention_mask','input_ids']]

%%time
results2 = []
for x in range(0, len(df_bp), 100000):
    print(len(df_bp)-x)
    df_test = df_bp.copy()
    df_test = df_test.iloc[x: x+100000]
    results2.append(df_test.apply(shorten_data_with_windows_after_tokenization, axis =1))

short_results2 = [i for info in results2 for i in info]
df_new = df_bp.copy()
df_new['short_results'] = short_results2
df_new = df_new.explode('short_results')

new_ner_tags = []
new_tokens = []
new_attention = []
new_input_ids = []
for x in df_new.short_results.to_list():
    if x['new_tags'] == []:
        new_ner_tags.append('None')
        new_tokens.append('None')
        new_attention.append('None')
        new_input_ids.append('None')
    else:
        if len(x['new_tags']) == len(x['new_tokens']) == len(x['new_attn']) == len(x['new_input_ids']):
            new_ner_tags.append(x['new_tags'])
            new_tokens.append(x['new_tokens'])
            new_attention.append(x['new_attn'])
            new_input_ids.append(x['new_input_ids'])
        else:
            new_ner_tags.append('None')
            new_tokens.append('None')
            new_attention.append('None')
            new_input_ids.append('None')

df_new['new_tags'] = new_ner_tags
df_new['new_tokens'] = new_tokens
df_new['new_attn'] = new_attention
df_new['new_input_ids'] = new_input_ids

df_new = df_new[df_new.new_tokens != 'None']
df_new = df_new.reset_index(drop=True)
df = df.drop(columns = ['short_text'])
df = df.rename(columns = {'short_text2':'short_text'})
print(df_new.cwetype.value_counts())

df_new['len_set_parts'] = df_new['parts'].apply(lambda x: len(set(x)))
df_bp_50 = df_new.copy()
df_bp_50 = df_bp_50[df_bp_50.parts.str.len() >= 50]
df_bp_50['parts_stripped'] = df_bp_50['parts'].apply(lambda x: x.strip())
df_bp_50['len_set_parts'] = df_bp_50['parts_stripped'].apply(lambda x: len(set(x)))
df_bp_50 = df_bp_50[df_bp_50.len_set_parts >= 5]
print(df_bp_50.cwetype.value_counts())

df_bp_50['len_set_toks'] = df_bp_50['new_tokens'].apply(lambda x: len(x))

df_final = df_bp_50.copy()

#saving just in case
import pickle
print('saving tagsandtokens df')
with open('df_tagsandtokens.pickle', 'wb') as output:
    pickle.dump(df_final, output)
    
df_final = df_final.drop(columns = ['ner_len','len_set_parts', 'parts_stripped'])

count = Counter()
for row in df_final.new_tags.tolist():
    for item in row:
        count[item] += 1
print(count.most_common(13))

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
with open(tokenizer_type.lower()+'_tagsandtokens.pickle', 'wb') as output:
    pickle.dump(dataset_v2, output)