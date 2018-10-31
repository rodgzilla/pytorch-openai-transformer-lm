import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from model_pytorch import DoubleHeadModel, load_openai_pretrained_model, DEFAULT_CONFIG
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import ClassificationLossCompute

n_updates = 0
best_score = 0

def load_dataset(path, train_size = 0.8, shuffle = True, seed = 42):
    df                             = pd.read_csv(path)#.sample(1000)
    X                              = df.comment
    y                              = df.sentiment
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        stratify     = y,
        train_size   = train_size,
        shuffle      = shuffle,
        random_state = seed
    )

    return (list(X_train), y_train.values), (list(X_val), y_val.values)

def transform_imdb(X, encoder, max_len, n_vocab, n_special, n_ctx):
    n_batch   = len(X)
    xmb       = np.zeros((n_batch, n_ctx, 2), dtype = np.int32)
    mmb       = np.zeros((n_batch, n_ctx), dtype = np.float32)
    start     = encoder['_start_']
    clf_token = encoder['_classify_']
    for i, x in enumerate(X):
        x_with_tokens   = [start] + x[:max_len] + [clf_token]
        l_x             = len(x_with_tokens)
        xmb[i, :l_x, 0] = x_with_tokens
        mmb[i, :l_x]    = 1
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)

    return xmb, mmb

def iter_apply(dh_model, n_batch_train, device, compute_loss_fct, Xs, Ms, Ys):
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost

def run_epoch(dh_model, n_batch_train, device, compute_loss_fct,
              logger, save_dir, desc, submit, n_valid, n_epochs,
              X_train, X_train_mask, y_train, X_val, X_val_mask,
              y_val):
    for xmb, mmb, ymb in iter_data(X_train,
                                   X_train_mask,
                                   y_train,
                                   n_batch = n_batch_train,
                                   truncate=True,
                                   verbose=True):
        global n_updates
        dh_model.train()
        XMB                    = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB                    = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB                    = torch.tensor(mmb).to(device)
        lm_logits, clf_logits  = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates             += 1
        if n_updates != 0 and n_updates % 5000 == 0:
            log(
                dh_model,
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

def log(dh_model, n_batch_train, device, compute_loss_fct, logger,
        save_dir, desc, submit, n_valid, n_epochs, n_updates, X_train,
        X_train_mask, y_train, X_val, X_val_mask, y_val):
    global best_score
    print("\nLogging")
    tr_logits, tr_cost = iter_apply(
        dh_model,
        n_batch_train,
        device,
        compute_loss_fct,
        X_train[:n_valid],
        X_train_mask[:n_valid],
        y_train[:n_valid]
    )
    va_logits, va_cost = iter_apply(
        dh_model,
        n_batch_train,
        device,
        compute_loss_fct,
        X_val,
        X_val_mask,
        y_val
    )
    tr_cost = tr_cost / len(y_train[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc  = accuracy_score(y_train[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc  = accuracy_score(y_val, np.argmax(va_logits, 1)) * 100.
    logger.log(
        n_epochs  = n_epochs,
        n_updates = n_updates,
        tr_cost   = tr_cost,
        va_cost   = va_cost,
        tr_acc    = tr_acc,
        va_acc    = va_acc
    )
    print('\n%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def main(epochs = 2):
    device                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args                  = DEFAULT_CONFIG
    encoder_path          = 'model/encoder_bpe_40000.json'
    bpe_path              = 'model/vocab_40000.bpe'
    save_dir              = 'save/'
    log_dir               = 'log/'
    desc                  = 'imdb_sentiment'
    submit                = True
    logger                = ResultLogger(
        path = os.path.join(
            log_dir,
            '{}.jsonl'.format(desc)
        ),
        **args.__dict__
    )
    text_encoder          = TextEncoder(encoder_path, bpe_path)
    encoder               = text_encoder.encoder
    (X_train, y_train), (X_val, y_val) = load_dataset(
        path       = 'data/aclImdb/train.csv',
        train_size = 0.9
    )
    (X_train, y_train), (X_val, y_val) = encode_dataset(
        (X_train, y_train),
        (X_val, y_val),
        encoder = text_encoder
    )
    n_batch_train               = 2
    n_train                     = len(y_train)
    n_valid                     = len(y_val) // 10
    n_updates_total             = (n_train // n_batch_train) * epochs
    n_vocab                     = len(encoder)
    encoder['_start_']          = len(encoder)
    encoder['_classify_']       = len(encoder)
    clf_token                   = encoder['_classify_']
    n_special                   = 2
    n_ctx                       = 512
    total_vocab_size            = n_vocab + n_special + n_ctx # positions also have an embedding
    X_train_trans, X_train_mask = transform_imdb(
        X_train,
        encoder,
        n_ctx - n_special,
        n_vocab,
        n_special,
        n_ctx
    )
    X_val_trans, X_val_mask     = transform_imdb(
        X_val,
        encoder,
        n_ctx - n_special,
        n_vocab,
        n_special,
        n_ctx
    )
    dh_model                    = DoubleHeadModel(
        args,
        clf_token,
        ('classification', 2),
        total_vocab_size,
        n_ctx
    )
    criterion                   = nn.CrossEntropyLoss(reduce = False)
    model_opt                   = OpenAIAdam(
        params        = dh_model.parameters(),
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
    compute_loss_fct            = ClassificationLossCompute(
        lm_criterion  = criterion,
        clf_criterion = criterion,
        lm_coef       = 0.5,
        opt           = model_opt
    )
    load_openai_pretrained_model(
        model     = dh_model.transformer,
        n_ctx     = n_ctx,
        n_special = n_special
    )
    dh_model.to(device)
    for epoch in range(epochs):
        run_epoch(
            dh_model,
            n_batch_train,
            device,
            compute_loss_fct,
            logger,
            save_dir,
            desc,
            submit,
            n_valid,
            epoch,
            X_train_trans,
            X_train_mask,
            y_train,
            X_val_trans,
            X_val_mask,
            y_val
        )
        log(
            dh_model,
            2 * n_batch_train,
            device,
            compute_loss_fct,
            logger,
            save_dir,
            desc,
            submit,
            n_valid,
            epoch,
            n_updates,
            X_train_trans,
            X_train_mask,
            y_train,
            X_val_trans,
            X_val_mask,
            y_val
        )

if __name__ == '__main__':
    main(5)
