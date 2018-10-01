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
    model.eval()
    lm_head.eval()
    for _ in range(max_size):
        # X, mmb   = encode_sentence(text_encoder, sentence, n_ctx)
        # model    = Model(args, vocab = vocab, n_ctx = n_ctx)
        # load_openai_pretrained_model(model, n_ctx = n_ctx, n_special = n_special)
        new_word = try_on_a_sentence(model, lm_head, text_encoder, sentence, n_ctx)
        sentence = f'{sentence} {new_word}'
        print('\n', sentence)

    return model

sentence = "Paul David Hewson, KBE OL (born 10 May 1960), known by his stage name Bono, is an Irish singer-songwriter, musician, venture capitalist, businessman, and philanthropist.[1] He is best known as the lead vocalist and primary lyricist of rock band U2"
# sentence = "Once upon a time"
model = main(sentence, 50)


# def main():
#     # xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)    encoder_path = 'model/encoder_bpe_40000.json'
#     encoder_path  = 'model/encoder_bpe_40000.json'
#     bpe_path      = 'model/vocab_40000.bpe'
#     text_encoder  = TextEncoder(encoder_path, bpe_path)
#     args          = DEFAULT_CONFIG
#     device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     sentence      = "I'm doing fine personally, what about "
#     X, mmb, n_ctx = encode_sentence(text_encoder, sentence)
#     model         = Model(args, n_ctx = n_ctx)
#     load_openai_pretrained_model(model)
#     lm_head       = LMHead(model, args)
#     try_on_a_sentence(model, lm_head, text_encoder, sentence)

#     return model


# def encode_sentence(encoder, sentence):
#     result     = encoder.encode([sentence])
#     n_ctx      = len(result[0])
#     n_vocab    = len(encoder.encoder)
#     X          = np.zeros((1, n_ctx, 2), dtype = np.int32)
#     mmb        = np.ones((1, n_ctx), dtype = np.float32)
#     X[0, :, 0] = result[0]
#     X[0, :, 1] = np.arange(n_vocab, n_vocab + n_ctx)

#     return X, mmb, n_ctx

# def main(max_size):
#     encoder_path           = 'model/encoder_bpe_40000.json'
#     bpe_path               = 'model/vocab_40000.bpe'
#     text_encoder           = TextEncoder(encoder_path, bpe_path)
#     encoder                = text_encoder.encoder
#     print(type(text_encoder))
#     encoder['_start_']     = len(encoder)
#     encoder['_delimiter_'] = len(encoder)
#     encoder['_classify_']  = len(encoder)
#     n_special              = 3
#     n_ctx                  = 512
#     args                   = DEFAULT_CONFIG
#     device                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     sentence = "Paul David Hewson, KBE OL (born 10 May 1960), known by his stage name Bono, is an Irish singer-songwriter, musician, venture capitalist, businessman, and philanthropist.[1] He is best known as the lead vocalist and primary lyricist of rock band U2"
#     # sentence               = "I'm doing fine personally, what about "
#     print(sentence)
#     for _ in range(max_size):
#         X, mmb, n_ctx = encode_sentence(text_encoder, sentence)
#         model         = Model(args, n_ctx = n_ctx)
#         load_openai_pretrained_model(model)
#         lm_head       = LMHead(model, args)
#         new_word      = try_on_a_sentence(model, lm_head, text_encoder, sentence)
#         sentence      = f'{sentence} {new_word}'
#         print(sentence)

#     return model
# def test_encoding():
#     text_encoder = TextEncoder('model/encoder_bpe_40000.json', 'model/vocab_40000.bpe')
#     print('benoit')
#     s = ["Hey everybody, what's going on? Are you okay or what?"]
#     encoded_sequences = text_encoder.encode(s)
#     decoded_sequences = [[text_encoder.decoder[i] for i in sequence] for sequence in encoded_sequences]
#     print(s)
#     print(encoded_sequences)
#     print(decoded_sequences)
#     encode_dataset_input = [[["Hey everybody, what's going on? Are you okay or what?"]]]
#     encoded_dataset = encode_dataset(encode_dataset_input, encoder = text_encoder)
#     print(encoded_dataset)
#     tensor = torch.tensor(encoded_dataset, dtype = torch.long)
#     print(tensor.shape)

# def test_loading_rocstories():
#     stories = rocstories('data/')
#     print(len(stories))
#     print(len(stories[0]))
#     print(len(stories[0][0]))
#     print(len(stories[0][0][0]))
#     print(type(stories))
#     print(type(stories[0]))
#     print(type(stories[0][0]))
#     print(type(stories[0][0][0]))
#     print(stories[0][0][0])
#     print(stories[0][0][1])
#     print(stories[0][0][2])
#     print(stories[0][0][3])
#     print(stories[0][0][4])
# test_loading_rocstories()
# test_encoding()
# sentence = 'Hi everybody, how are'
# text_encoder = TextEncoder('model/encoder_bpe_40000.json', 'model/vocab_40000.bpe')
# encode_sentence(text_encoder, sentence)
