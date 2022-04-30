import random
import codecs
import joblib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy

from CW_attack import CarliniL2
from util import logger, root_dir, args
import models
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import BERTScorer


class YelpDataset(Dataset):
    def __init__(self, path_or_raw, raw=False):
        self.raw = raw
        if not self.raw:
            self.data = joblib.load(path_or_raw)
        else:
            self.max_len = 512
            self.data = path_or_raw
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            for data in self.data:
                data['seq'] = tokenizer.encode(('[CLS] ' + data['adv_text']))
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]

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


def transform(seq, unk_words_dict=None):
    if unk_words_dict is None:
        unk_words_dict = {}
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    unk_count = 0
    for x in seq:
        if x == 100:
            unk_count += 1
    if unk_count == 0 or len(unk_words_dict) == 0:
        return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])
    else:
        tokens = []
        for idx, x in enumerate(seq):
            if x == 100 and len(unk_words_dict[idx]) != 0:
                unk_words = unk_words_dict[idx]
                unk_word = random.choice(unk_words)
                tokens.append(unk_word)
            else:
                tokens.append(tokenizer._convert_id_to_token(x))
        return tokenizer.convert_tokens_to_string(tokens)


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


def get_knowledge_dict(input_knowledge_dict):
    knowledge_dict = {0: [0], 101: [101]}
    for k, v in input_knowledge_dict.items():
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


def get_bug_dict(input_bug_dict, input_ids):
    bug_dict = {0: [0], 101: [101]}
    unk_words_dict = {}
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    unk_cnt = 0
    for x in input_ids:
        if x == 100:
            unk_cnt += 1
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in bug_dict:
            for j in range(len(token_list)):
                if input_ids[i] == input_ids[j]:
                    if j in unk_words_dict:
                        unk_words_dict[i] = unk_words_dict[j]
                    break
            continue
        word = token_list[i]
        if word not in input_bug_dict:
            bug_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_bug_dict[word]
        unk_id = 100
        unk_list = []
        for unk_word in [x[0] for x in candidates if tokenizer._convert_token_to_id(x[0]) == unk_id]:
            adv_seq = deepcopy(token_list)
            adv_seq[i] = unk_word
            adv_seq = tokenizer.encode(tokenizer.convert_tokens_to_string(adv_seq))
            adv_unk_cnt = 0
            for x in adv_seq:
                if x == 100:
                    adv_unk_cnt += 1
            if adv_unk_cnt == unk_cnt + 1:
                unk_list = [unk_word]
                break
        unk_words_dict[i] = unk_list
        candidates = [tokenizer._convert_token_to_id(x[0]) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        if len(unk_list) == 0:
            while 100 in candidates:
                candidates.remove(100)
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        bug_dict[input_ids[i]] = candidates

    return bug_dict, unk_words_dict


def cw_word_attack(data_val):
    logger.info("Begin Attack")
    logger.info(("const confidence:", args.const, args.confidence))

    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    adv_pickle = []

    test_batch = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
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
            attack_targets = batch['label']
        else:
            if args.strategy == 0:
                if batch['label'][0] == 1:
                    attack_targets = torch.full_like(batch['label'], 0)
                else:
                    attack_targets = torch.full_like(batch['label'], 1)
            elif args.strategy == 1:
                if batch['label'][0] < 2:
                    attack_targets = torch.full_like(batch['label'], 4)
                else:
                    attack_targets = torch.full_like(batch['label'], 0)
        label = batch['label'] = batch['label'].to(device)
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

        if args.function == 'all':
            cluster_char_dict = get_similar_dict(batch['similar_dict'])
            bug_char_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['seq'][0])
            similar_char_dict = get_knowledge_dict(batch['knowledge_dict'])

            for k, v in cluster_char_dict.items():
                synset = list(set(v + similar_char_dict[k]))
                while 100 in synset:
                    synset.remove(100)
                if len(synset) >= 1:
                    similar_char_dict[k] = synset
                else:
                    similar_char_dict[k] = [k]

            for k, v in bug_char_dict.items():
                synset = list(set(v + similar_char_dict[k]))
                # while 100 in synset:
                #     synset.remove(100)
                if len(synset) >= 1:
                    similar_char_dict[k] = synset
                else:
                    similar_char_dict[k] = [k]

            all_dict = similar_char_dict
        elif args.function == 'typo':
            all_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['seq'][0])
        elif args.function == 'knowledge':
            all_dict = get_knowledge_dict(batch['knowledge_dict'])
            unk_words_dict = None
        elif args.function == 'cluster':
            all_dict = get_similar_dict(batch['similar_dict'])
            unk_words_dict = None
        else:
            raise Exception('Unknown perturbation function.')

        cw.wv = all_dict
        cw.mask = cw_mask
        cw.seq = data
        cw.batch_info = batch
        cw.seq_len = seq_len

        # attack
        adv_data = cw.run(model, input_embedding, attack_targets)
        # retest
        adv_seq = torch.tensor(batch['seq']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    adv_seq.data[bi, i] = all_dict[adv_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]

        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += torch.sum((prediction != label).float()).item()
        tot += len(batch['label'])

        for i in range(len(batch['label'])):
            diff = difference(adv_seq[i], data[i])
            adv_pickle.append({
                'index': batch_index,
                'adv_text': transform(adv_seq[i], unk_words_dict),
                'orig_text': transform(batch['seq'][i]),
                'raw_text': batch['raw_text'][i],
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
                if batch_index % 100 == 0:
                    try:
                        # logger.info(("label:", label[i].item()))
                        # logger.info(("pred:", prediction[i].item()))
                        # logger.info(("ori_pred:", ori_prediction[i].item()))
                        # logger.info(("target:", attack_targets[i].item()))
                        # logger.info(("orig:", transform(batch['seq'][i])))
                        # logger.info(("adv:", transform(adv_seq[i], unk_words_dict)))
                        # logger.info(("seq_len:", batch['seq_len'][i].item()))

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
    logger.info(("const confidence:", args.const, args.confidence))


def check_consistency():
    logger.info("Start checking consistency")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '')

    adv_text = YelpDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text, batch_size=args.batch_size, shuffle=False)

    inconsistent = []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_batch)):
            batch['seq'] = torch.stack(batch['seq']).t().to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            out = model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            if pred[0].item() != batch['pred'][0]:
                inconsistent.append((bi, batch))

    logger.info("Num of inconsistent: {}".format(len(inconsistent)))
    if len(inconsistent) != 0:
        joblib.dump(inconsistent, os.path.join(root_dir, 'inconsistent_adv.pkl'))

    return adv_text


def validate():
    logger.info("Start validation")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '')

    adv_text_dataset = YelpDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text_dataset, batch_size=args.batch_size, shuffle=False)

    test_model = models.BertC(dropout=args.dropout, num_class=5)
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
            orig_text_eval.append(item['orig_text'].replace('[CLS]', '').strip())
            adv_text_eval.append(item['adv_text'].replace('[CLS]', '').strip())
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
    logger.info("Start attack")
    model = models.BertC(dropout=args.dropout, num_class=5)
    try:
        model.load_state_dict(torch.load(args.load, map_location=torch.device('cuda')))
    except RuntimeError:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.load, map_location=torch.device('cuda')).items()})
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    test_data = YelpDataset(args.test_data)

    cw_word_attack(test_data)
    validate()
