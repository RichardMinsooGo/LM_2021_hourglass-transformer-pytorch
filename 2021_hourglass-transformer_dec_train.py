'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
# !pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm

'''
D1. Import Libraries for Data Engineering
'''
import csv
import sys
import os
import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from IPython.display import display

# Setup seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
D3. [PASS] Tokenizer Install & import
'''
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN  = 15
DECODER_LEN  = 23
BATCH_SIZE   = 16

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', None)

"""
raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),

    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),

    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
    ("I have something very important to tell you.", "Il me faut vous dire quelque chose de très important."),
    ("I have three times as many books as he does.", "J'ai trois fois plus de livres que lui."),
    ("I have to change the batteries in the radio.", "Il faut que je change les piles de cette radio."),
    ("I have to finish up some things before I go.", "Je dois finir deux trois trucs avant d'y aller."),

    ("I have to think about what needs to be done.", "Je dois réfléchir sur ce qu'il faut faire."),
    ("I haven't been back here since the incident.", "Je ne suis pas revenu ici depuis l'accident."),
    ("I haven't eaten anything since this morning.", "Je n'ai rien mangé depuis ce matin."),
    ("I hear his business is on the verge of ruin.", "Apparemment son entreprise est au bord de la faillite."),
    ("I hope I didn't make you feel uncomfortable.", "J'espère que je ne t'ai pas mis mal à l'aise."),
    ("I hope to continue to see more of the world.", "J'espère continuer à voir davantage le monde."),
    ("I hope to see reindeer on my trip to Sweden.", "J'espère voir des rennes lors de mon voyage en Suède."),
    ("I hope you'll find this office satisfactory.", "J'espère que ce bureau vous conviendra."),

    ("I hurried in order to catch the first train.", "Je me dépêchai pour avoir le premier train."),
    ("I just can't stand this hot weather anymore.", "Je ne peux juste plus supporter cette chaleur."),
    ("I just don't want there to be any bloodshed.", "Je ne veux tout simplement pas qu'il y ait une effusion de sang."),
    ("I just thought that you wouldn't want to go.", "J'ai simplement pensé que vous ne voudriez pas y aller."),
    ("I plan to go. I don't care if you do or not.", "Je prévois d'y aller. Ça m'est égal que vous y alliez aussi ou pas."),
    ("I prefer soap as a liquid rather than a bar.", "Je préfère le savon liquide à une savonnette."),
    ("I promise you I'll explain everything later.", "Je vous promets que j'expliquerai tout plus tard."),
    ("I ran as fast as I could to catch the train.", "Je courus aussi vite que je pus pour attraper le train."))


raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"))
"""

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"))

import unicodedata
import re

from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
def preprocess_en(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def preprocess_fr(sentence):
    # 위에서 구현한 함수를 내부적으로 호출
    sentence = unicode_to_ascii(sentence.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1", sentence)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sentence = re.sub(r"[^a-zA-Z!.?]+", r" ", sentence)

    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

# 인코딩 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous deja dine?"

print(preprocess_en(en_sent))
print(preprocess_fr(fr_sent).encode('utf-8'))

raw_encoder_input, raw_data_fr = list(zip(*raw_data))
raw_encoder_input, raw_data_fr = list(raw_encoder_input), list(raw_data_fr)

raw_src = [preprocess_en(data) for data in raw_encoder_input]
raw_trg = [preprocess_fr(data) for data in raw_data_fr]

print(raw_src[:4])
print(raw_trg[:4])

'''
D9. Define dataframe
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
total_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(total_df)) # 리뷰 개수 출력
total_df.sample(3)

raw_src_df  = total_df['SRC']
raw_trg_df  = total_df['TRG']

src_sentence  = raw_src_df
trg_sentence  = raw_trg_df

'''
D10. Define tokenizer
'''

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(total_df['TRG']))

# This is the folder to save the data. Modify it to suit your environment.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"

vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
Token List
'''
# Recommend : For small number of vocabulary, please test each IDs.
# src_vocab_list
src_vocab_list = [[vocab_src.id_to_piece(id), id] for id in range(vocab_src.get_piece_size())]

# trg_vocab_list
trg_vocab_list = [[vocab_trg.id_to_piece(id), id] for id in range(vocab_trg.get_piece_size())]

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [  SRC_df.iloc[1,0],  SRC_df.iloc[2,0],  SRC_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_src.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_src.DecodeIds(txt_2_ids))

    txt_2_tkn = vocab_src.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_src.DecodePieces(txt_2_tkn))

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_src.id_to_piece(ids2))
    print("\n")

print("\n")

# Target Tokenizer
lines = [  TRG_df.iloc[1,0],  TRG_df.iloc[2,0],  TRG_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_trg.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_trg.DecodeIds(txt_2_ids))
    
    txt_2_tkn = vocab_trg.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_trg.DecodePieces(txt_2_tkn))

    ids2 = vocab_trg.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_trg.id_to_piece(ids2))
    print("\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_src  = vocab_src.encode_as_ids(src_sentence.to_list())
tokenized_trg  = vocab_trg.encode_as_ids(trg_sentence.to_list())

# Add [BOS], [EOS] token ids to each target list elements.
new_list = [ x.insert(0, 2) for x in tokenized_trg]
new_list = [ x.insert(len(x), 3) for x in tokenized_trg]

tokenized_inputs  = tokenized_src
tokenized_outputs = tokenized_trg

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

'''
D15. Send data to device
'''

tensors_src   = torch.tensor(tkn_sources).to(device)
tensors_trg   = torch.tensor(tkn_targets).to(device)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. [PASS] Split Data
'''

'''
D18. Build dataset
'''

from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

dataset    = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import math

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else ((val,) * depth)

# factory

def get_hourglass_transformer(
    dim,
    *,
    depth,
    shorten_factor,
    attn_resampling,
    updown_sample_type,
    **kwargs
):
    assert isinstance(depth, int) or (isinstance(depth, tuple)  and len(depth) == 3), 'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)'
    assert not (isinstance(depth, int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return Transformer(dim = dim, depth = depth, **kwargs)

    return HourglassTransformer(dim = dim, depth = depth, shorten_factor = shorten_factor, attn_resampling = attn_resampling, updown_sample_type = updown_sample_type, **kwargs)

# up and down sample classes

class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, 'b (n s) d -> b n d', 'mean', s = self.shorten_factor)

class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'b n d -> b (n s) d', s = self.shorten_factor)

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s = self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s = self.shorten_factor)

# classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None):
        h, device = self.heads, x.device
        kv_input = default(context, x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            mask = rearrange(mask, 'i j -> () () i j')
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# transformer classes

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        norm_out = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, causal = causal)),
                PreNormResidual(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, context = None, mask = None):
        for attn, ff in self.layers:
            x = attn(x, context = context, mask = mask)
            x = ff(x)

        return self.norm(x)

class HourglassTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        shorten_factor = 2,
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False
    ):
        super().__init__()
        assert len(depth) == 3, 'depth should be a tuple of length 3'
        assert updown_sample_type in {'naive', 'linear'}, 'downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)'

        pre_layers_depth, valley_depth, post_layers_depth = depth

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head
        )

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample   = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample   = LinearUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.valley_transformer = get_hourglass_transformer(
            shorten_factor = rest_shorten_factor,
            depth = valley_depth,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            causal = causal,
            **transformer_kwargs
        )

        self.attn_resampling_pre_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None
        self.attn_resampling_post_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None

        self.pre_transformer = Transformer(depth = pre_layers_depth, causal = causal, **transformer_kwargs)
        self.post_transformer = Transformer(depth = post_layers_depth, causal = causal, **transformer_kwargs)
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, mask = None):
        # b : batch, n : sequence length, d : feature dimension, s : shortening factor

        s, b, n = self.shorten_factor, *x.shape[:2]

        # top half of hourglass, pre-transformer layers

        x = self.pre_transformer(x, mask = mask)

        # pad to multiple of shortening factor, in preparation for pooling

        x = pad_to_multiple(x, s, dim = -2)

        if exists(mask):
            padded_mask = pad_to_multiple(mask, s, dim = -1, value = False)

        # save the residual, and for "attention resampling" at downsample and upsample

        x_residual = x.clone()

        # if autoregressive, do the shift by shortening factor minus one

        if self.causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value = 0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value = False)

        # naive average pool

        downsampled = self.downsample(x)

        if exists(mask):
            downsampled_mask = reduce(padded_mask, 'b (n s) -> b n', 'sum', s = s) > 0
        else:
            downsampled_mask = None

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled

        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = s)
            else:
                attn_resampling_mask = None

            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = b)

        # the "valley" - either a regular transformer or another hourglass

        x = self.valley_transformer(downsampled, mask = downsampled_mask)

        valley_out = x.clone()

        # naive repeat upsample

        x = self.upsample(x)

        # add the residual

        x = x + x_residual

        # post-valley "attention resampling"

        if exists(self.attn_resampling_post_valley):
            x = self.attn_resampling_post_valley(
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                rearrange(valley_out, 'b n d -> (b n) () d')
            )

            x = rearrange(x, '(b n) s d -> b (n s) d', b = b)

        # bring sequence back to original length, if it were padded for pooling

        x = x[:, :n]

        # post-valley transformers

        x = self.post_transformer(x, mask = mask)
        return self.norm_out(x)

# main class

class HourglassTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        shorten_factor = None,
        heads = 8,
        dim_head = 64,
        attn_resampling = True,
        updown_sample_type = 'naive',
        causal = True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = get_hourglass_transformer(
            dim = dim,
            depth = depth,
            shorten_factor = shorten_factor,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            dim_head = dim_head,
            heads = heads,
            causal = causal,
            norm_out = True
        )

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, mask = None):
        device = x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        x = self.transformer(x, mask = mask)
        return self.to_logits(x)

import torch
# from hourglass_transformer_pytorch import HourglassTransformerLM

model = HourglassTransformerLM(
    num_tokens = n_dec_vocab,               # number of tokens
    dim = 512,                      # feature dimension
    max_seq_len = 1024,             # maximum sequence length
    heads = 8,                      # attention heads
    dim_head = 64,                  # dimension per attention head
    shorten_factor = 2,             # shortening factor
    depth = (4, 2, 4),              # tuple of 3, standing for pre-transformer-layers, valley-transformer-layers (after downsample), post-transformer-layers (after upsample) - the valley transformer layers can be yet another nested tuple, in which case it will shorten again recursively
)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/Hourglass_transformer.pt'):
    model.load_state_dict(torch.load('./checkpoints/Hourglass_transformer.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    accuracies = []

    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:
        for batch_idx, samples in enumerate(dataloader):
            src_inputs, trg_outputs = samples

            # print("src_inputs  Shape :", src_inputs.shape)
            # print(src_inputs)
            mask_src = (src_inputs!=0).int()
            # print(mask_src)

            # print("trg_outputs Shape :", trg_outputs.shape)
            # print("trg_outputs :\n", trg_outputs)
            mask_trg = (trg_outputs!=0).int()
            # print(mask_trg)

            Input_concat = torch.concat((src_inputs, trg_outputs),dim=1)
            # print("Input_concat Shape :", Input_concat.shape)
            # print("Input_concat :\n", Input_concat)

            with torch.set_grad_enabled(True):

                # Transformer에 입력
                logits_lm = model(Input_concat)
                # print("logits_lm  Shape :", logits_lm.shape)
                
                pad       = torch.LongTensor(trg_outputs.size(0), 1).fill_(0).to(device)
                preds_id  = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)
                # print("labels_lm Shape: \n",labels_lm.shape)
                # print("labels_lm : \n",labels_lm)

                labels_concat = torch.concat((src_inputs, labels_lm),dim=1)
                # print("labels_concat Shape :", labels_concat.shape)
                # print("labels_concat :\n", labels_concat)
                
                optimizer.zero_grad()
                loss = criterion(preds_id, labels_concat)  # loss 계산

                # Accuracy
                # print("preds_id  : \n",preds_id.shape)
                mask_0 = (labels_concat!=0).int()
                arg_preds_id = torch.argmax(preds_id, axis=1)
                # print("arg_preds : \n",arg_preds_id)
                # print("arg_preds : \n",arg_preds_id.shape)
                # print("mask_0    : \n",mask_0)

                accuracy_1 = torch.eq(labels_concat, arg_preds_id).int()
                # print("accuracy_1 : \n",accuracy_1)

                accuracy_2 = torch.mul(arg_preds_id, accuracy_1).int()
                # print("accuracy_2 : \n",accuracy_2)

                accuracy = torch.count_nonzero(accuracy_2) / torch.count_nonzero(mask_0)
                # print("Accuracy : ",accuracy.clone().detach().cpu().numpy())
                accuracies.append(accuracy.clone().detach().cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
    print("accuracies :", np.mean(accuracies))
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

N_EPOCHS = 100

for epoch in range(N_EPOCHS):

    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)

    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/Hourglass_transformer.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Build evaluation code.

# Predict the trained model
trained_model = HourglassTransformerLM(
    num_tokens = n_dec_vocab,               # number of tokens
    dim = 512,                      # feature dimension
    max_seq_len = 1024,             # maximum sequence length
    heads = 8,                      # attention heads
    dim_head = 64,                  # dimension per attention head
    shorten_factor = 2,             # shortening factor
    depth = (4, 2, 4),              # tuple of 3, standing for pre-transformer-layers, valley-transformer-layers (after downsample), post-transformer-layers (after upsample) - the valley transformer layers can be yet another nested tuple, in which case it will shorten again recursively
).to(device)
trained_model.load_state_dict(torch.load('./checkpoints/Hourglass_transformer.pt'))

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(text):
    text = preprocess_sentence(text)
    # print(text)
    text = [vocab_src.encode_as_ids(text)]
    # print(text)
    encoder_input = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')
    # print(encoder_input)

    decoder_input = [2]   #[BOS] token is 2
    # print(decoder_input)
    
    input  = torch.tensor(encoder_input).to(device)
    output = torch.tensor([decoder_input]).to(device)

    # print("input :", input)
    # print("output:", output)

    for i in range(DECODER_LEN):
        concate_input = torch.concat((input, output),dim=1)
        # print("concate_input :", concate_input)
        predictions = trained_model(concate_input)
        # print(predictions)

        predictions = predictions[:, -1:, :]
        # print(predictions)

        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions, axis=-1)
        # print(predicted_id)
        if predicted_id== 3:
            break

        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)[0].detach().cpu().numpy()
    prediction = prediction[1:]
    # print("Pred IDs :", prediction)

    predicted_sentence = vocab_trg.DecodeIds(prediction.tolist())
    # print(predicted_sentence)
    return predicted_sentence

for idx in (0, 1, 2, 3):
    print("Input        :", src_sentence[idx])
    print("Prediction   :", predict(src_sentence[idx]))
    print("Ground Truth :", trg_sentence[idx],"\n")


'''
M13. [PASS] Explore the training result with test dataset
'''
    
