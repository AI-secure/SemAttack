from __future__ import print_function
from CW_attack import CarliniL2
from models import *
from my_generator.dataset import YelpDataset
from util import logger, get_args, root_dir

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os
from tqdm import tqdm
from vocab import Vocab

import spacy

import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import BERTScorer


def cal_ppl(text, model, tokenizer):
    assert isinstance(text, list)
    encodings = tokenizer('\n\n'.join(text), return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()

    return ppl


def cal_bert_score(cands, refs, scorer):
    _, _, f1 = scorer.score(cands, refs)
    return f1.mean()


def transform(seq):
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    return vocab.convertToLabels(seq)


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1

    return tot


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
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


def init_attack():
    for param in model.parameters():
        param.requires_grad = False


def get_similar_dict(similar_dict, text):
    similar_char_dict = {}
    for k, v in similar_dict.items():
        k = vocab.convertToIdx([k], UNK_WORD)[0]
        v = [vocab.convertToIdx([x], UNK_WORD)[0] for x in v]
        if k not in v:
            v.append(k)
        while 0 in v:
            v.remove(0)
        while 1 in v:
            v.remove(1)
        while 2 in v:
            v.remove(2)
        while 3 in v:
            v.remove(3)
        if len(v) >= 1:
            similar_char_dict[k] = v
        else:
            similar_char_dict[k] = [k]
    for char_ind in text:
        if char_ind not in similar_char_dict.keys():
            similar_char_dict[char_ind] = [char_ind]

    return similar_char_dict


def get_knowledge_dict(knowledge_dict, text):
    similar_char_dict = {}
    for k, v in knowledge_dict.items():
        k = vocab.convertToIdx([k], UNK_WORD)[0]
        v = [vocab.convertToIdx([x], UNK_WORD)[0] for x in v]
        if k not in v:
            v.append(k)
        while 0 in v:
            v.remove(0)
        while 1 in v:
            v.remove(1)
        while 2 in v:
            v.remove(2)
        while 3 in v:
            v.remove(3)
        if len(v) >= 1:
            similar_char_dict[k] = v
        else:
            similar_char_dict[k] = [k]
    for char_ind in text:
        if char_ind not in similar_char_dict.keys():
            similar_char_dict[char_ind] = [char_ind]

    return similar_char_dict


def get_bug_dict(input_bug_dict, text):
    bug_dict = {}
    for k, v in input_bug_dict.items():
        k = vocab.convertToIdx([k], UNK_WORD)[0]
        v = [vocab.convertToIdx([x], UNK_WORD)[0] for x in v]
        if k not in v:
            v.append(k)
        while 0 in v:
            v.remove(0)
        # while 1 in v:
        #     v.remove(1)
        while 2 in v:
            v.remove(2)
        while 3 in v:
            v.remove(3)
        v = list(set(v))
        if len(v) >= 1:
            bug_dict[k] = v
        else:
            bug_dict[k] = [k]
    for char_ind in text:
        if char_ind not in bug_dict.keys():
            bug_dict[char_ind] = [char_ind]

    return bug_dict


def get_batch(data_val, has_tree=False):
    for batch, i in enumerate(tqdm(range(0, len(data_val), args.batch_size))):
        batch_data = data_val[i:min(i + args.batch_size, len(data_val))]
        text_len = [len(x['text']) for x in batch_data]
        text_len = [500 if x > 500 else x for x in text_len]
        text = [x['text'] for x in batch_data]
        # split_text = [x['split_text'] for x in batch_data]
        label = [x['label'] for x in batch_data]
        data, targets = package(text, label)

        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()

        # find the most attention sentence
        model.encoder.raw_inp = None
        model.encoder.bilstm.attack_mode = False
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_correct = torch.sum((prediction == targets).float())

        # append it to the front of the paragraph
        batch_add_start = []
        batch_add_end = []

        for i, seq in enumerate(text_len):
            batch_add_start.append(0)
            batch_add_end.append(seq)

        similar_char_dict = [get_similar_dict(x['similar_dict'], x['text']) for x in batch_data]
        bug_dict = [get_bug_dict(x['bug_dict'], x['text']) for x in batch_data]
        knowledge_dict = [get_knowledge_dict(x['knowledge_dict'], x['text']) for x in batch_data]

        attack_targets = [4 if x < 2 else 0 for x in label]
        data, attack_targets = package(text, attack_targets)
        if args.cuda:
            data = data.cuda()
            attack_targets = attack_targets.cuda()

        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        orig_append_correct = torch.sum((prediction == targets).float())

        result = {'data': data, 'targets': targets, 'add_start': batch_add_start, 'text': text, 'text_len': text_len,
                  'add_end': batch_add_end, 'label': label, 'orig_correct': orig_correct, 'ori_pred': prediction,
                  'orig_append_correct': orig_append_correct, 'attack_targets': attack_targets,
                  'similar_char_dict': similar_char_dict, 'bug_dict': bug_dict, 'knowledge_dict': knowledge_dict}

        yield batch, result


def cw_word_attack(data_val):
    logger.info("Begin Attack")
    logger.info(("const confidence:", args.const, args.confidence))
    init_attack()
    fname = "full-vectors.kv"
    if not os.path.isfile(fname):
        embed = model.encoder.bilstm.encoder.weight
        print(len(vocab.idxToLabel), embed.shape[1], file=open(fname, "a"))
        for k, v in vocab.idxToLabel.items():
            vectors = embed[k].cpu().numpy()
            vector = ""
            for x in vectors:
                vector += " " + str(x)
            print(v, vector[1:], file=open(fname, "a"))
    device = torch.device("cuda:0" if args.cuda else "cpu")
    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    adv_pickle = []

    cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=True)
    for batch_index, batch in get_batch(data_val):
        if batch['ori_pred'][0].item() != batch['label'][0]:
            continue
        data = batch['data']
        if args.untargeted:
            attack_targets = torch.LongTensor(batch['label']).cuda()
        else:
            attack_targets = batch['attack_targets']
        batch_add_start = batch['add_start']
        batch_add_end = batch['add_end']
        text = batch['text']
        label = batch['label']
        cluster_char_dict = batch['similar_char_dict'][0]
        bug_dict = batch['bug_dict'][0]
        knowledge_dict = batch['knowledge_dict'][0]
        # convert text into embedding and attack in the embedding space
        model.encoder.raw_inp = data
        model.init_hidden(data.size(1))
        model.encoder.bilstm.attack_mode = True
        input_embedding = model.encoder.bilstm.encoder(data)

        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            cw_mask[add_start:add_end, bi] = 1
        cw_mask = torch.from_numpy(cw_mask).float()

        if args.function == 'all':
            for k, v in cluster_char_dict.items():
                if k in knowledge_dict.keys():
                    synset = list(set(v + knowledge_dict[k]))
                else:
                    synset = v
                if len(synset) >= 1:
                    knowledge_dict[k] = synset
                else:
                    knowledge_dict[k] = [k]

            for k, v in bug_dict.items():
                if k in knowledge_dict.keys():
                    synset = list(set(v + knowledge_dict[k]))
                else:
                    synset = v
                if len(synset) >= 1:
                    knowledge_dict[k] = synset
                else:
                    knowledge_dict[k] = [k]

            all_dict = knowledge_dict
        elif args.function == 'typo':
            all_dict = bug_dict
        elif args.function == 'knowledge':
            all_dict = knowledge_dict
        elif args.function == 'cluster':
            all_dict = cluster_char_dict
        else:
            raise Exception('Unknown perturbation function.')

        if args.cuda:
            cw_mask = cw_mask.cuda()
            cw.batch_info = batch
            cw.wv = all_dict

        cw.mask = cw_mask
        adv_data = cw.run(model, input_embedding, attack_targets)

        adv_seq = torch.tensor(data).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    adv_seq.data[i, bi] = all_dict[adv_seq.data[i, bi].item()][cw.o_best_sent[bi][i - add_start]]

        model.encoder.raw_inp = None
        model.encoder.bilstm.attack_mode = False
        output, attention = model(adv_seq)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]

        targets = batch['targets']
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == targets).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += torch.sum((prediction != targets).float()).item()
        tot += len(label)

        for i in range(len(text)):
            diff = difference(adv_seq[:, i], text[i])
            adv_pickle.append({
                'index': batch_index,
                'adv_text': transform(adv_seq[:, i]),
                'orig_text': transform(text[i]),
                'label': label[i],
                'target': attack_targets[i].item(),
                'ori_pred': batch['ori_pred'][i].item(),
                'pred': prediction[i].item(),
                'diff': diff,
                'orig_seq': text[i],
                'adv_seq': adv_seq[:, i].cpu().numpy().tolist(),
                'seq_len': batch['text_len'][i]
            })
            assert batch['ori_pred'][i].item() == label[i]
            if (args.untargeted and prediction[i].item() != label[i]) or (not args.untargeted and prediction[i].item() == attack_targets[i].item()):
                tot_diff += diff
                tot_len += batch['text_len'][i]
                if batch_index % 100 == 0:
                    try:
                        # logger.info(("label:", label[i]))
                        # logger.info(("pred:", prediction[i].item()))
                        # logger.info(("ori_pred:", batch['ori_pred'][i].item()))
                        # logger.info(("target:", attack_targets[i].item()))
                        # logger.info(("orig:", transform(text[i])))
                        # logger.info(("adv:", transform(adv_seq[:, i])))
                        # logger.info(("seq_len:", batch['text_len'][i]))

                        logger.info(("tot:", tot))
                        logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
                        logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
                        logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
                        logger.info(("orig_correct: {:.1f}%".format(orig_correct / tot * 100)))
                        logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
                        if args.untargeted:
                            logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                        else:
                            logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                    except:
                        continue
    joblib.dump(adv_pickle, os.path.join(root_dir, 'adv_text.pkl'))
    logger.info(("tot:", tot))
    logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
    logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
    logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
    logger.info(("orig_correct: {:.1f}%".format(orig_correct / tot * 100)))
    logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
    if args.untargeted:
        logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    else:
        logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    logger.info(("const confidence lr:", args.const, args.confidence, args.lr))


def validate():
    tokenizer = spacy.load('en_core_web_sm')
    logger.info("Start validation")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        words = tokenizer(' '.join(i['adv_text']))
        i['adv_text_tokens'] = list(map(lambda x: x.text.lower(), words))

    for item in adv_text:
        item['adv_text_tokens'] = vocab.convertToIdx(item['adv_text_tokens'], UNK_WORD)

    test_model = Classifier({
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
        test_model = test_model.cuda()
    test_model.load_state_dict(torch.load(args.test_model))
    if args.cuda:
        test_model = test_model.cuda()
    test_model.eval()

    with torch.no_grad():
        for batch, i in enumerate(tqdm(range(0, len(adv_text), args.batch_size))):
            batch_data = adv_text[i:i + args.batch_size]
            text = [x['adv_text_tokens'] for x in batch_data]
            label = [x['label'] for x in batch_data]
            target = [x['target'] for x in batch_data]
            data, labels = package(text, label)
            if args.cuda:
                data = data.cuda()
                labels = labels.cuda()
            hidden = test_model.init_hidden(data.size(1))
            output, attention = test_model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            prediction = torch.max(output_flat, 1)[1]
            adv_text[i]['pred_validated'] = prediction[0].item()

    joblib.dump(adv_text, os.path.join(root_dir, 'adv_text_validated.pkl'))

    acc = 0
    origin_success = 0
    total = 0
    total_change = 0
    total_word = 0
    orig_text_eval = []
    adv_text_eval = []
    for item in tqdm(adv_text):
        if item['ori_pred'] != item['label']:
            origin_success += 1
            continue
        if args.untargeted and item['pred_validated'] != item['label'] or not args.untargeted and item['pred_validated'] == item['target']:
            acc += 1
            total_change += item['diff']
            total_word += item['seq_len']
            orig_text_eval.append(' '.join(item['orig_text']))
            adv_text_eval.append(' '.join(item['adv_text']))
        total += 1

    global model
    del model
    del test_model
    torch.cuda.empty_cache()

    model_id = 'gpt2-large'
    ppl_model = GPT2LMHeadModel.from_pretrained(model_id).to('cuda')
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    bs_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    orig_ppl = cal_ppl(orig_text_eval, ppl_model, ppl_tokenizer)
    adv_ppl = cal_ppl(adv_text_eval, ppl_model, ppl_tokenizer)
    bs = cal_bert_score(adv_text_eval, orig_text_eval, bs_scorer)

    suc = float(acc / total) * 100
    change_rate = float(total_change / total_word) * 100
    origin_acc = (1 - origin_success / len(adv_text)) * 100

    logger.info(sys.argv)
    logger.info('orig acc：{:.1f}%'.format(origin_acc))
    logger.info('attack success：{:.1f}'.format(acc))
    logger.info('orig pred success：{:.1f}'.format(total))
    logger.info('success rate：{:.1f}%'.format(suc))
    logger.info('change rate: {:.1f}%'.format(change_rate))
    logger.info('original ppl: {:.1f}'.format(orig_ppl))
    logger.info('adv ppl: {:.1f}'.format(adv_ppl))
    logger.info('bert score: {:.3f}'.format(bs))


if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Load Dictionary
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
    print('Begin to load data.')
    import joblib

    if args.load:
        model.load_state_dict(torch.load(args.load))
        if args.cuda:
            model = model.cuda()

        data_val = YelpDataset(args.test_data, vocab)
        # cw_word_attack(data_val)
        validate()
