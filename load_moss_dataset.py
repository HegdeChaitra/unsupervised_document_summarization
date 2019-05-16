import pandas as pd
import numpy as np
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle

import pdb
from torch.utils.data import DataLoader

from torch import optim
import time

UNK_IDX = 2
PAD_IDX = 3
SOS_token = 0
EOS_token = 1

def read_dataset(file):
    f = open(file)
    list_l = []
    for line in f:
        list_l.append(line.strip())
    df = pd.DataFrame()
    df['data'] = list_l
    return df

class Lang:
    def __init__(self, name, minimum_count = 5):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = ["SOS","EOS","UKN","PAD"]
        self.n_words = 4  # Count SOS and EOS
        self.minimum_count = minimum_count

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())


    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        if self.word2count[word] >= self.minimum_count:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
    #             self.index2word[self.n_words] = word
                self.index2word.append(word)
                self.n_words += 1
            
            
def split(df):
    df['en_tokenized'] = df["en_data"].apply(lambda x:x.lower().split( ))
#     df['vi_tokenized'] = df['vi_data'].apply(lambda x:x.lower().split( ))
    return df



def token2index_dataset(df,en_lang,vi_lang):
#     for lan in ['en','vi']:
    indices_data = []

    lang_obj = en_lang

    for tokens in df['en' +'_tokenized']:
        index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
        index_list.append(EOS_token)
#             index_list.insert(0,SOS_token)
        indices_data.append(index_list)
    df['en'+'_idized'] = indices_data
    return df


def test_load(MAX_LEN, old_lang_obj, path):
       
    print("Reading English: test")
#     en_all = read_dataset(path+"corpus.tc.en")
#     en_val = read_dataset(path+"newstest2015.tc.en")
    
    # Load in gigawords
    df = pd.read_csv('../../../scratch/brs426/Moss_test.csv', header=0)
    print(df.columns.values)
    
    # Load in language object
    if old_lang_obj:
        with open(old_lang_obj, 'rb') as f:
            en_lang = pickle.load(f)
            vi_lang = pickle.load(f)
            
    
#     print("Reading CS: training and validation and test")
#     en_ru = read_dataset(path+"corpus.tc.cs")
#     cs_val = read_dataset(path+"newstest2015.tc.cs")
#     cs_test = read_dataset(path+'newstest2016.tc.cs')
#     print("Done")
    
#     df = pd.DataFrame()
#     df['en_data'] = en_all['data']
#     df['vi_data'] = en_ru['data']
    
#     print("Shuffling and Splitting")
#     df = df.sample(n=500000).reset_index(drop=True)
    
#     train = df
    #len_df = len(df)
    #train_len = int(len_df * 0.95)
    
    #train = df.iloc[:train_len , :]
    #val = df.iloc[train_len: , :]
    
#     print(train.head(2))
#     print(val.head(2))
#     print(test.head(2))
#     train = pd.DataFrame()
#     train['en_data'] = en_train['data']
#     train['vi_data'] = vi_train['data']
    
#     val = pd.DataFrame()
#     val['en_data'] = en_val['data']
#     val['vi_data'] = cs_val['data']
    
    test = pd.DataFrame()
    test['en_data'] = df['input.txt']
    
#     test['vi_data'] = cs_test['data']
    
    if old_lang_obj:
        with open(old_lang_obj,'rb') as f:
            en_lang = pickle.load(f)
            vi_lang = pickle.load(f)
#     else:
#         print("creating language objects")
#         en_lang = Lang("en")
#         for ex in train['en_data']:
#             en_lang.addSentence(ex)
    
#         vi_lang = Lang("vi")
#         for ex in train['vi_data']:
#             vi_lang.addSentence(ex)
        
#         with open("lang_obj_new.pkl",'wb') as f:
#             pickle.dump(en_lang, f)
#             pickle.dump(vi_lang, f)
            
    print("tokenizing")
#     train = split(train)
#     val = split(val)
    test = split(test)
    
    print("token to index mapping")
#     train = token2index_dataset(train,en_lang,vi_lang)
#     val = token2index_dataset(val,en_lang,vi_lang)
    test = token2index_dataset(test,en_lang,vi_lang)
    
    print("Computing length")
#     train['en_len'] = train['en_idized'].apply(lambda x: len(x))
#     train['vi_len'] = train['vi_idized'].apply(lambda x:len(x))
    
#     val['en_len'] = val['en_idized'].apply(lambda x: len(x))
#     val['vi_len'] = val['vi_idized'].apply(lambda x: len(x))
    
    test['en_len'] = test['en_idized'].apply(lambda x: len(x))
    test['ref0'] = df['task1_ref0']
    test['ref1'] = df['task1_ref1']
    test['ref2'] = df['task1_ref2']
    test['ref3'] = df['task1_ref3']
    test['ref4'] = df['task1_ref4']
#     test['vi_len'] = test['vi_idized'].apply(lambda x: len(x))
    
#     train = train[np.logical_and(train['en_len']>=2,train['vi_len']>=2)]
#     train = train[train['vi_len']<=MAX_LEN]
    
#     val = val[np.logical_and(val['en_len']>=2,val['vi_len']>=2)]
#     val = val[val['vi_len']<=MAX_LEN]
    
    return test, en_lang, vi_lang
