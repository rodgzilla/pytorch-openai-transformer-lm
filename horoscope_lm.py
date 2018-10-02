import pandas as pd
import pdb
import argparse
import itertools

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pytorch import TransformerModel, LMHead, load_openai_pretrained_model, DEFAULT_CONFIG
from model_pytorch import LanguageModel
from utils import encode_dataset, flatten, iter_data, ResultLogger, make_path
from text_utils import TextEncoder
from opt import OpenAIAdam

def _chunk_word_list(word_list, max_sequence_len = 50000):
    # We have to split the text into text of 100.000 characters
    # because of the parser limitations.
    word_sequences    = [[]]
    last_sequence_len = 0
    for word in word_list:
        # If the last word list has reached the maximum size
        if last_sequence_len + len(word) > max_sequence_len:
            # We transform it into a string by rejoining the words
            word_sequences[-1] = ' '.join(word_sequences[-1])
            # and then begin a new word sequence
            word_sequences.append([])
            last_sequence_len = 0
        word_sequences[-1].append(word)
        last_sequence_len += len(word)

    if type(word_sequences[-1]) == list:
        word_sequences[-1] = ' '.join(word_sequences[-1])

    return word_sequences

# TODO: Add the shuffle code
def load_dataset(text_encoder, window_size, path = 'data/horoscope_dataset.csv',
                 shuffle = True, seed = 142857,
                 test_size = 0.8):
    df             = pd.read_csv(path)
    all_text       = ' '.join(df.TEXT)
    word_list      = all_text.split(' ')
    word_sequences = _chunk_word_list(word_list, )
    encoded_text   = text_encoder.encode(word_sequences)
    word_idx_list  = list(itertools.chain.from_iterable(encoded_text))
    context_list   = []
    target_list    = []

    for start_idx in range(len(word_idx_list) - window_size - 1):
        context_list.append(word_idx_list[start_idx : start_idx + window_size])
        target_list.append(word_idx_list[start_idx + window_size])

    X_train, X_val, y_train, y_val = train_test_split(
        context_list,
        target_list,
        test_size    = test_size,
        shuffle      = shuffle,
        random_state = seed
    )
    return (X_train, y_train), (X_val, y_val)

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

def transform_dataset(dataset, encoder, max_len, n_vocab, n_special, n_ctx):
    n_batch   = len(dataset)
    xmb       = np.zeros((n_batch, n_ctx, 2), dtype = np.int32)
    mmb       = np.zeros((n_batch, n_ctx), dtype = np.float32)
    start     = encoder.encoder['_start_']
    clf_token = encoder.encoder['_classify_']
    for i, x in enumerate(dataset):
        x_with_tokens   = [start] + x[:max_len] + [clf_token]
        l_x             = len(x_with_tokens)
        xmb[i, :l_x, 0] = x_with_tokens
        mmb[i, :l_x]    = 1
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)

    return xmb, mmb

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

epochs                             = 3
n_batch_train                      = 16
args                               = DEFAULT_CONFIG
window_size                        = 80
max_len                            = window_size
bpe_path                           = 'model/vocab_40000.bpe'
encoder_path                       = 'model/encoder_bpe_40000.json'
text_encoder                       = TextEncoder(encoder_path, bpe_path)
encoder                            = text_encoder.encoder
n_vocab                            = len(encoder)
encoder['_start_']                 = len(encoder)
encoder['_classify_']              = len(encoder)
clf_token                          = encoder['_classify_']
n_special                          = 2
n_ctx                              = window_size + n_special
total_vocab_size                   = n_vocab + n_special + n_ctx
(X_train, y_train), (X_val, y_val) = load_dataset(
    text_encoder,
    window_size = window_size,
    path        = 'data/small_horoscope_dataset.csv'
)
n_train                     = len(y_train)
n_updates_total             = (n_train // n_batch_train) * epochs
X_train_trans, X_train_mask = transform_dataset(
    X_train,
    text_encoder,
    window_size,
    n_vocab,
    n_special,
    n_ctx
)
X_val_trans, X_val_mask = transform_dataset(
    X_val,
    text_encoder,
    window_size,
    n_vocab,
    n_special,
    n_ctx
)
language_model = LanguageModel(
    args,
    vocab = total_vocab_size,
    n_ctx = n_ctx
)
load_openai_pretrained_model(
    model.transformer,
    n_ctx = n_ctx,
    n_special = n_special
)
model_opt = OpenAIAdam(
    params = language_model.parameters(),
    lr            = 6.25e-5,
    schedule      = 'warmup_linear',
    warmup        = 0.002,
    t_total       = n_updates_total,
    b1            = 0.9,
    b2            = 0.999,
    e             = 1e-8,
    l2            = 0.01,
    vector_l2     = 'store_true',
    max_grad_norm = 1
)
