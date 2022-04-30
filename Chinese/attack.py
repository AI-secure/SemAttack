import random
import codecs
import joblib
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from CW_attack import CarliniL2
from util import logger, root_dir, args
import models
from pytorch_transformers import BertTokenizer


class ThuNewsDataset(Dataset):
    def __init__(self, path_or_raw, raw=False):
        self.raw = raw
        if not self.raw:
            self.data = joblib.load(path_or_raw)
        else:
            self.max_len = 30
            self.data = path_or_raw
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            for data in self.data:
                data['seq'] = tokenizer.encode(('[CLS] ' + data['adv_text']))
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]
                data['seq_len'] = len(data['seq'])

    def __len__(self):
        if not self.raw:
            return len(self.data) // args.scale
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not self.raw:
            return self.data[index * args.scale]
        else:
            return self.data[index]


def transform(seq):
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1
    return tot


def get_similar_dict(similar_dict):
    similar_char_dict = {0: [0], 101: [101]}
    for k, v in similar_dict.items():
        k = tokenizer._convert_token_to_id(k)
        v = [tokenizer._convert_token_to_id(x[0]) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            similar_char_dict[k] = v
        else:
            similar_char_dict[k] = [k]

    return similar_char_dict


def get_knowledge_dict(input_dict):
    knowledge_dict = {0: [0], 101: [101]}
    for k, v in input_dict.items():
        if len(k) == 1:
            k = tokenizer._convert_token_to_id(k)
            v = [tokenizer._convert_token_to_id(x[0]) for x in v]
            if k not in v:
                v.append(k)
            while 100 in v:
                v.remove(100)
            if len(v) >= 1:
                knowledge_dict[k] = v
            else:
                knowledge_dict[k] = [k]

    return knowledge_dict


def get_knowledge_dict_only(input_dict):
    knowledge_dict = {0: [0], 101: [101]}
    for k, v in input_dict.items():
        k = tokenizer.encode(k)
        v = [tokenizer.encode(x[0]) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            knowledge_dict[tuple(k)] = v
        else:
            knowledge_dict[tuple(k)] = [k]

    return knowledge_dict


def cw_word_attack(data_val):
    logger.info("Begin Attack")
    logger.info(("const confidence target:", args.const, args.confidence, args.target))

    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    adv_pickle = []

    test_batch = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    if args.function == 'knowledge':
        cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=True, word_mode=True)
    else:
        cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=True)
    for batch_index, batch in enumerate(tqdm(test_batch)):
        batch_add_start = batch['add_start'] = []
        batch_add_end = batch['add_end'] = []
        for i, seq in enumerate(batch['seq_len']):
            batch['add_start'].append(1)
            batch['add_end'].append(seq)

        data = batch['seq'] = torch.stack(batch['seq']).t().to(device)
        orig_sent = transform(batch['seq'][0])

        seq_len = batch['seq_len'] = batch['seq_len'].to(device)
        if args.untargeted:
            attack_targets = batch['class']
        else:
            if args.strategy == 0:
                # if batch['class'][0] == 1:
                #     attack_targets = torch.full_like(batch['class'], 0)
                # else:
                #     attack_targets = torch.full_like(batch['class'], 1)
                attack_targets = torch.full_like(batch['class'], args.target)
            elif args.strategy == 1:
                if batch['class'][0] == 13:
                    attack_targets = torch.full_like(batch['class'], 9)
                else:
                    attack_targets = torch.full_like(batch['class'], 13)
        label = batch['class'] = batch['class'].to(device)
        attack_targets = attack_targets.to(device)

        # test original acc
        out = model(batch['seq'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        if ori_prediction[0].item() != label[0].item():
            continue
        batch['orig_correct'] = torch.sum((prediction == label).float())

        # prepare attack
        input_embedding = model.bert.embeddings.word_embeddings(data)
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for i, seq in enumerate(batch['seq_len']):
            cw_mask[i][1:seq] = 1

        # attack
        try:
            if args.function == 'all':
                similar_char_dict = get_similar_dict(batch['typo_dict'])
                cluster_char_dict = get_similar_dict(batch['similar_dict'])
                knowledge_dict = get_knowledge_dict(batch['knowledge_dict'])

                for k, v in cluster_char_dict.items():
                    synset = list(set(v + similar_char_dict[k]))
                    while 100 in synset:
                        synset.remove(100)
                    if len(synset) >= 1:
                        similar_char_dict[k] = synset
                    else:
                        similar_char_dict[k] = [k]

                for k, v in knowledge_dict.items():
                    synset = list(set(v + similar_char_dict[k]))
                    while 100 in synset:
                        synset.remove(100)
                    if len(synset) >= 1:
                        similar_char_dict[k] = synset
                    else:
                        similar_char_dict[k] = [k]

                all_dict = similar_char_dict
            elif args.function == 'knowledge':
                all_dict = get_knowledge_dict_only(batch['knowledge_dict'])
            elif args.function == 'cluster':
                all_dict = get_similar_dict(batch['similar_dict'])
            elif args.function == 'typo':
                all_dict = get_similar_dict(batch['typo_dict'])
            else:
                raise Exception('Unknown perturbation function.')

            cw.wv = all_dict
            cw.mask = cw_mask
            cw.seq = data
            cw.batch_info = batch
            cw.seq_len = seq_len

            adv_data = cw.run(model, input_embedding, attack_targets)
        except:
            continue

        # retest
        adv_seq = torch.tensor(batch['seq']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    if args.function == 'knowledge':
                        if batch['start_mark'][i] == 1:
                            if i == len(batch['start_mark']) - 1:
                                j = i + 1
                            else:
                                j = batch['start_mark'].index(1, i + 1)
                            adv_seq.data[bi, i:j] = torch.LongTensor(
                                all_dict[tuple(adv_seq.data[bi, i:j].cpu().numpy())][
                                    cw.o_best_sent[bi][sum(batch['start_mark'][add_start:i])]]).cuda()
                    else:
                        adv_seq.data[bi, i] = all_dict[adv_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]

        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        tot += len(batch['class'])
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += torch.sum((prediction != label).float()).item()

        for i in range(len(batch['class'])):
            diff = difference(adv_seq[i], data[i])
            adv_pickle.append({
                'index': batch_index,
                'adv_text': transform(adv_seq[i]),
                'orig_text': transform(batch['seq'][i]),
                'label': label[i].item(),
                'target': attack_targets[i].item(),
                'ori_pred': ori_prediction[i].item(),
                'pred': prediction[i].item(),
                'diff': diff,
                'orig_seq': batch['seq'][i].cpu().numpy().tolist(),
                'adv_seq': adv_seq[i].cpu().numpy().tolist(),
                'seq_len': batch['seq_len'][i].item()
            })
            assert ori_prediction[i].item() == label[i].item()
            if (args.untargeted and prediction[i].item() != label[i].item()) or (not args.untargeted and prediction[i].item() == attack_targets[i].item()):
                tot_diff += diff
                tot_len += batch['seq_len'][i].item()
                try:
                    logger.info(("untargeted:", args.untargeted))
                    logger.info(("label:", label[i].item()))
                    logger.info(("pred:", prediction[i].item()))
                    logger.info(("ori_pred:", ori_prediction[i].item()))
                    logger.info(("target:", attack_targets[i].item()))
                    logger.info(("orig:", transform(batch['seq'][i])))
                    logger.info(("adv:", transform(adv_seq[i])))
                    logger.info(("seq_len:", batch['seq_len'][i].item()))

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
    logger.info(("const confidence target:", args.const, args.confidence, args.target))


def validate():
    logger.info("Start validation")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '').replace('[PAD]', '').replace(' ', '')

    adv_text_dataset = ThuNewsDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text_dataset, batch_size=args.batch_size, shuffle=False)

    test_model = models.BertC(dropout=args.dropout, num_class=14)
    test_model.load_state_dict(torch.load(args.test_model, map_location=torch.device('cuda')))
    test_model = test_model.to(device)
    test_model.eval()

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_batch)):
            batch['seq'] = torch.stack(batch['seq']).t().to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            out = test_model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            adv_text[bi]['pred_validated'] = pred[0].item()

    joblib.dump(adv_text, os.path.join(root_dir, 'adv_text_validated.pkl'))

    acc = 0
    origin_success = 0
    total = 0
    total_change = 0
    total_word = 0
    for item in tqdm(adv_text):
        if item['ori_pred'] != item['label']:
            origin_success += 1
            continue
        if args.untargeted and item['pred_validated'] != item['label'] or not args.untargeted and item['pred_validated'] == item['target']:
            acc += 1
            total_change += item['diff']
            total_word += item['seq_len']
        total += 1

    suc = float(acc / total) * 100
    change_rate = float(total_change / total_word) * 100
    origin_acc = (1 - origin_success / len(adv_text)) * 100

    logger.info('orig acc：{:.1f}%'.format(origin_acc))
    logger.info('attack success：{:.1f}'.format(acc))
    logger.info('orig pred success：{:.1f}'.format(total))
    logger.info('success rate：{:.1f}%'.format(suc))
    logger.info('change rate: {:.1f}%'.format(change_rate))


if __name__ == '__main__':
    logger.info("Start attack")
    model = models.BertC(dropout=args.dropout, num_class=14)
    model.load_state_dict(torch.load(args.load, map_location=torch.device('cuda')))
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    test_data = ThuNewsDataset(args.test_data)
    cw_word_attack(test_data)
    validate()
