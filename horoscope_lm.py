import os
import pandas as pd
import pdb
import argparse
import itertools

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pytorch import TransformerModel, LMHead, load_openai_pretrained_model, DEFAULT_CONFIG
from model_pytorch import LanguageModel
from utils import encode_dataset, flatten, iter_data, ResultLogger, make_path
from text_utils import TextEncoder
from opt import OpenAIAdam
from loss import LanguageModelingLossCompute

n_updates  = 0
best_score = 0

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

def iter_apply(model, n_batch_train, device, compute_loss_fct, Xs, Ms, Ys, return_logits = True):
    if return_logits:
        logits = []
    cost = 0
    with torch.no_grad():
        model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            lm_logits = model(XMB)
            lm_logits *= n
            lm_losses = compute_loss_fct(XMB, YMB, MMB, lm_logits, only_return_losses=True)
            lm_losses *= n
            if return_logits:
                logits.append(lm_logits.to("cpu").numpy())
            cost += lm_losses.sum().item()

    if return_logits:
        logits = np.concatenate(logits, 0)
        return logits, cost

    return cost

def run_epoch(model, n_batch_train, device, compute_loss_fct, logger,
              save_dir, desc, submit, n_valid, n_epochs, X_train,
              X_train_mask, y_train, X_val, X_val_mask, y_val):
    for xmb, mmb, ymb in iter_data(X_train,
                                   X_train_mask,
                                   y_train,
                                   n_batch = n_batch_train,
                                   truncate=True,
                                   verbose=True):
        global n_updates
        model.train()
        XMB        = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB        = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB        = torch.tensor(mmb).to(device)
        lm_logits  = model(XMB)
        compute_loss_fct(XMB, YMB, MMB, lm_logits)
        n_updates += 1
        if n_updates != 0 and n_updates % 10 == 0:
            log(
                model,
                n_batch_train,
                device,
                compute_loss_fct,
                logger,
                save_dir,
                desc,
                submit,
                n_valid,
                n_epochs,
                n_updates,
                X_train,
                X_train_mask,
                y_train,
                X_val,
                X_val_mask,
                y_val
            )

def log(model, n_batch_train, device, compute_loss_fct, logger,
        save_dir, desc, submit, n_valid, n_epochs, n_updates, X_train,
        X_train_mask, y_train, X_val, X_val_mask, y_val):
    global best_score
    print("\nLogging")
    tr_cost = iter_apply(
        model,
        n_batch_train,
        device,
        compute_loss_fct,
        X_train[:n_valid],
        X_train_mask[:n_valid],
        y_train[:n_valid],
        False
    )
    va_cost = iter_apply(
        model,
        n_batch_train,
        device,
        compute_loss_fct,
        X_val,
        X_val_mask,
        y_val,
        False
    )
    tr_cost = tr_cost / len(y_train[:n_valid])
    va_cost = va_cost / n_valid
    logger.log(
        n_epochs  = n_epochs,
        n_updates = n_updates,
        tr_cost   = tr_cost,
        va_cost   = va_cost
    )
    print('\n%d %d %.3f %.3f' % (n_epochs, n_updates, tr_cost, va_cost))
    if submit:
        score = va_cost
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(model.state_dict(), make_path(path))

# Training configuration
epochs                             = 3
n_batch_train                      = 2
window_size                        = 89
max_len                            = window_size
# General configuration
save_dir                           = 'save/'
log_dir                            = 'log/'
desc                               = 'horoscope_language_model'
submit                             = True
args                               = DEFAULT_CONFIG
logger                             = ResultLogger(
    path = os.path.join(
        log_dir,
        '{}.jsonl'.format(desc)
    ),
    **args.__dict__
)
device                             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    path        = 'data/tiny_horoscope_dataset.csv'
)
n_train                     = len(y_train)
n_valid                     = len(y_val) // 10
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
    language_model.transformer,
    n_ctx = n_ctx,
    n_special = n_special
)
language_model.to(device)
model_opt = OpenAIAdam(
    params        = language_model.parameters(),
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
criterion        = nn.CrossEntropyLoss(reduce = False)
compute_loss_fct = LanguageModelingLossCompute(
    lm_criterion = criterion,
    opt = model_opt
)

for epoch in range(epochs):
    run_epoch(
        model            = language_model,
        n_batch_train    = n_batch_train,
        device           = device,
        compute_loss_fct = compute_loss_fct,
        logger           = logger,
        save_dir         = save_dir,
        desc             = desc,
        submit           = submit,
        n_valid          = n_valid,
        n_epochs         = epoch,
        X_train          = X_train_trans,
        X_train_mask     = X_train_mask,
        y_train          = y_train,
        X_val            = X_val_trans,
        X_val_mask       = X_val_mask,
        y_val            = y_val
    )
