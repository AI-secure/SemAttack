from __future__ import print_function
from models import *

from util import get_args

import torch
import torch.nn as nn
import torch.optim as optim

import json
import numpy as np
import time
import random
import os
from tqdm import tqdm
from vocab import Vocab
from tensorboardX import SummaryWriter


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
        # return torch.sum((torch.sum(torch.sum((mat ** 2), 2), 1)) ** 0.5) / mat.shape[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def package(texts, targets):
    """Package data for training / evaluation."""
    maxlen = 0
    for item in texts:
        maxlen = max(maxlen, len(item))
    maxlen = min(maxlen, 500)
    for i in range(len(texts)):
        if maxlen < len(texts[i]):
            texts[i] = texts[i][:maxlen]
        else:
            for j in range(maxlen - len(texts[i])):
                texts[i].append(vocab.getIndex('<pad>'))
    texts = torch.LongTensor(texts)
    targets = torch.LongTensor(targets)
    return texts.t(), targets

def get_next_trade_day(date:str, prices):
    import datetime as dt
    from datetime import datetime
    date = datetime.strptime(date, "%m%d%Y")
    max_date = datetime.strptime('10012018', "%m%d%Y")
    while date not in prices:
        date = date + dt.timedelta(days=1)
        if date >= max_date:
            return None

        dstr = datetime.strftime(date, "%m%d%Y")
        if dstr in prices:
            return dstr


def evaluate(data_val):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(data_val), args.batch_size)):
            # data, targets = package(data_val['text'][i:min(len(data_val['text']), i + args.batch_size)],
            #                         data_val['label'][i:min(len(data_val['text']), i + args.batch_size)])
            batch_data = data_val[i:i + args.batch_size]
            text = [x['text'] for x in batch_data]
            label = [x['label'] for x in batch_data]
            data, targets = package(text, label)
            if args.cuda:
                data = data.cuda()
                targets = targets.cuda()
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += criterion(output_flat, targets).data
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float())
            # attention = torch.sum(attention, 1)
            # print(attention[0][:100])
    return total_loss.item() / (len(data_val) // args.batch_size), total_correct.item() / len(data_val)


def train(epoch_number):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    for batch, i in enumerate(tqdm(range(0, len(data_train), args.batch_size))):
        batch_data = data_train[i:i + args.batch_size]
        text = [x['text'] for x in batch_data]
        label = [x['label'] for x in batch_data]
        data, targets = package(text, label)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data
        if batch % 100 == 0:
            sum_attention = torch.sum(attention, 1)
            # print(sum_attention[0][:100])
        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += args.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            writer.add_scalar('train/loss', total_loss, batch * args.batch_size + epoch_number * len(data_train))
            writer.add_scalar('train/pure_loss', total_pure_loss, batch * args.batch_size + epoch_number * len(data_train))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()
    evaluate_start_time = time.time()
    train_loss, train_acc = evaluate(data_train)
    writer.add_scalar('eval_train/loss', train_loss, epoch_number)
    writer.add_scalar('eval_train/accuracy', train_acc, epoch_number)

    print('-' * 89)
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), train_loss, train_acc))
    print('-' * 89)

    evaluate_start_time = time.time()
    val_loss, acc = evaluate(data_val)
    print('-' * 89)
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
    print('-' * 89)
    writer.add_scalar('eval/loss', val_loss, epoch_number)
    writer.add_scalar('eval/accuracy', acc, epoch_number)

    # Save the model, if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model.state_dict(), f)
        f.close()
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:

        with open(args.save[:-3] + '.best_acc.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
        f.close()
        best_acc = acc
    with open(args.save[:-3] + '.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    writer = SummaryWriter()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.train_data)
    assert os.path.exists(args.val_data)
    print('Begin to load the dictionary.')

    PAD_WORD = '<pad>'
    UNK_WORD = '<unk>'
    EOS_WORD = '<eos>'
    SOS_WORD = '<sos>'
    vocab = Vocab(filename=args.dictionary, data=[PAD_WORD, UNK_WORD, EOS_WORD, SOS_WORD])

    best_val_loss = None
    best_acc = None

    n_token = vocab.size()
    model = Classifier({
        'dropout': args.dropout,
        'ntoken': n_token,
        'nlayers': args.nlayers,
        'nhid': args.nhid,
        'ninp': args.emsize,
        'pooling': 'all',
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'vocab': vocab,
        'word-vector': args.word_vector,
        'class-number': args.class_number
    })
    if args.cuda:
        model = model.cuda()

    print(args)
    I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
    for i in range(args.batch_size):
        for j in range(args.attention_hops):
            I.data[i][j][j] = 1
    if args.cuda:
        I = I.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=args.wd)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    import joblib

    # if args.load:
    #     model.load_state_dict(torch.load(args.load))
    #     if args.cuda:
    #         model = model.cuda()
    #     print('-' * 89)
    #     data_val = np.load(args.train_data)[()]
    #     evaluate_start_time = time.time()
    #     test_loss, acc = evaluate(data_val)
    #     print('-' * 89)
    #     fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
    #     print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
    #     print('-' * 89)
    #
    #     print('-' * 89)
    #     data_val = joblib.load(args.test_data)
    #     evaluate_start_time = time.time()
    #     test_loss, acc = evaluate(data_val)
    #     print('-' * 89)
    #     fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
    #     print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
    #     print('-' * 89)
    #     exit(0)

    data_train = joblib.load(args.train_data)
    data_val = joblib.load(args.val_data)
    try:
        for epoch in range(args.epochs):
            train(epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exit from training early.')
        data_val = joblib.load(args.test_data)
        evaluate_start_time = time.time()
        test_loss, acc = evaluate(data_val)
        print('-' * 89)
        fmt = '| test | time: {:5.2f}s | test loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), test_loss, acc))
        print('-' * 89)
        exit(0)

    writer.close()
