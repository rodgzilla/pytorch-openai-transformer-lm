import pandas as pd
import pdb
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pytorch import TransformerModel, LMHead, load_openai_pretrained_model, DEFAULT_CONFIG
from utils import encode_dataset, flatten, iter_data, ResultLogger, make_path
from text_utils import TextEncoder
from datasets import rocstories

def load_dataset(path = 'data/horoscope_dataset.csv', shuffle = True,
                 seed = 42):
    df = pd.read_csv(path)
    all_text = ' '.join(df.TEXT)
    word_list = all_text.split(' ')
    # We have to split the text into text of 100.000 characters
    # because of the parser limitations.
    word_sequences = [[]]
    last_sequence_len = 0
    max_sequence_len = 50000
    for word in word_list:
        if last_sequence_len + len(word) > max_sequence_len:
            word_sequences.append([])
            last_sequence_len = 0
        word_sequences[-1].append(word)
        last_sequence_len += len(word)

    return word_sequences

def try_on_a_sentence(model, lm_head, text_encoder, sentence, n_ctx):
    # pdb.set_trace()
    n_vocab            = len(text_encoder.encoder)
    # X, mmb           = encode_sentence(text_encoder, sentence, n_ctx)
    X, _, input_length = encode_sentence(text_encoder, sentence, n_ctx)
    X_tensor           = torch.tensor(X, dtype = torch.long)
    # mmb_tensor       = torch.tensor(mmb)
    transformer_output = model(X_tensor)
    lm_output          = lm_head(transformer_output)
    lm_output          = lm_output[:, :n_vocab]
    new_word_idx       = lm_output[input_length - 2].max(dim = 0)[1].item()
    # new_word           = text_encoder.decoder[lm_output.max(dim = 1)[1][-1].item()][:-4]
    new_word           = text_encoder.decoder[new_word_idx][:-4]

    return new_word

def encode_sentence(encoder, sentence, n_ctx):
    result                 = encoder.encode([sentence])
    n_vocab                = len(encoder.encoder)
    X                      = np.zeros((1, n_ctx, 2), dtype = np.int32)
    mmb                    = np.zeros((1, n_ctx), dtype = np.float32)
    start                  = encoder.encoder['_start_']
    clf_token              = encoder.encoder['_classify_']
    encoded_input          = [start] + result[0] + [clf_token]
    input_length           = len(encoded_input)
    X[0, :input_length, 0] = [start] + result[0] + [clf_token]
    X[0, :, 1]             = np.arange(n_vocab, n_vocab + n_ctx)
    mmb[0, :input_length]  = 1

    return X, mmb, input_length

def main(sentence, max_size):
    encoder_path           = 'model/encoder_bpe_40000.json'
    bpe_path               = 'model/vocab_40000.bpe'
    text_encoder           = TextEncoder(encoder_path, bpe_path)
    encoder                = text_encoder.encoder
    n_vocab                = len(text_encoder.encoder)
    encoder['_start_']     = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_']  = len(encoder)
    n_special              = 3
    n_ctx                  = 100
    vocab                  = n_vocab + n_special + n_ctx
    args                   = DEFAULT_CONFIG
    device                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model                  = TransformerModel(args, vocab = vocab, n_ctx = n_ctx)
    load_openai_pretrained_model(model, n_ctx = n_ctx, n_special = n_special)
    lm_head                = LMHead(model, args)
    # model.eval()
    # lm_head.eval()
    # for _ in range(max_size):
    #     # X, mmb   = encode_sentence(text_encoder, sentence, n_ctx)
    #     # model    = Model(args, vocab = vocab, n_ctx = n_ctx)
    #     # load_openai_pretrained_model(model, n_ctx = n_ctx, n_special = n_special)
    #     new_word = try_on_a_sentence(model, lm_head, text_encoder, sentence, n_ctx)
    #     sentence = f'{sentence} {new_word}'
    #     print('\n', sentence)

    return model

bpe_path       = 'model/vocab_40000.bpe'
encoder_path   = 'model/encoder_bpe_40000.json'
text_encoder   = TextEncoder(encoder_path, bpe_path)
word_sequences = load_dataset()
