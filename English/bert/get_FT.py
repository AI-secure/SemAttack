import random
import joblib
import torch
import string
from torch.utils.data import Dataset
from tqdm import tqdm

from util import logger, root_dir, args
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM


class YelpDataset(Dataset):
    def __init__(self, path):
        cache_path = 'FC_' + path
        save_path = 'FT_' + cache_path
        self.data = joblib.load(cache_path)
        bug_data = []
        for i, data in enumerate(tqdm(self.data)):
            data['bug_dict'] = get_bug_dict(data['seq'])
            bug_data.append(data)
            if i % 1000 == 0:
                joblib.dump(bug_data, save_path)
        joblib.dump(self.data, save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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


def bug_delete(word):
    res = word
    point = random.randint(1, len(word) - 2)
    res = res[0:point] + res[point + 1:]
    return res


def bug_swap(word):
    if len(word) <= 4:
        return word
    res = word
    points = random.sample(range(1, len(word) - 1), 2)
    a = points[0]
    b = points[1]

    res = list(res)
    w = res[a]
    res[a] = res[b]
    res[b] = w
    res = ''.join(res)
    return res


def bug_sub_C(word):
    res = word
    key_neighbors = get_key_neighbors()
    point = random.randint(0, len(word) - 1)

    if word[point] not in key_neighbors:
        return word
    choices = key_neighbors[word[point]]
    subbed_choice = choices[random.randint(0, len(choices) - 1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)

    return res


def bug_insert(word):
    if len(word) >= 6:
        return word
    res = word
    point = random.randint(1, len(word) - 1)
    res = res[0:point] + random.choice(string.ascii_lowercase) + res[point:]
    return res


def get_key_neighbors():
    # By keyboard proximity
    neighbors = {
        "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
        "i": "uojkl", "o": "ipkl", "p": "ol",
        "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
        "j": "yuihknm", "k": "uiojlm", "l": "opk",
        "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
    }
    # By visual proximity
    neighbors['i'] += '1'
    neighbors['l'] += '1'
    neighbors['z'] += '2'
    neighbors['e'] += '3'
    neighbors['a'] += '4'
    neighbors['s'] += '5'
    neighbors['g'] += '6'
    neighbors['b'] += '8'
    neighbors['g'] += '9'
    neighbors['q'] += '9'
    neighbors['o'] += '0'

    return neighbors


def get_bug(word):
    bugs = [word]
    if len(word) <= 2:
        return bugs
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    return list(set(bugs))


def get_bug_dict(indexed_tokens):
    bug_dict = {}
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(1, len(indexed_tokens)):
        if tokenized_words[i] in word_list:
            words = get_bug(tokenized_words[i])
        else:
            words = []
        if len(words) >= 1:
            bug_dict[tokenized_words[i]] = words
        else:
            bug_dict[tokenized_words[i]] = [tokenized_words[i]]

    return bug_dict


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    word_list = joblib.load(args.word_list)
    torch.manual_seed(args.seed)
    test_data = YelpDataset(args.test_data)
