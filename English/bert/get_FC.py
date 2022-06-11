import joblib
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from util import logger, root_dir, args
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

from collections import Counter


class YelpDataset(Dataset):
    def __init__(self, path):
        cache_path = 'FC_' + path
        self.max_len = 512
        if os.path.exists(cache_path):
            self.data = joblib.load(cache_path)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.data = joblib.load(path)
            clustered_data = []
            for i, data in enumerate(tqdm(self.data)):
                data['seq'] = tokenizer.encode(('[CLS] ' + data['raw_text']))
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]
                data['seq_len'] = len(data['seq'])
                data['similar_dict'] = get_similar_dict(data['seq'])
                clustered_data.append(data)
                if i % 100 == 0:
                    joblib.dump(clustered_data, cache_path)
            joblib.dump(self.data, cache_path)

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


def get_knn(t, k):
    dist = torch.norm(embedding_space - t, dim=1, p=None)
    knn = dist.topk(k, largest=False)
    words = []
    for index in knn.indices:
        words.append(word_list[index])
    count = Counter(words)
    sorted_words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_words(words, neighbors):
    words = [item[0] for item in words if item[1] >= neighbors]
    return words


def get_similar_dict(indexed_tokens):
    similar_char_dict = {}
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    mask_tensor = torch.tensor([[1] * len(indexed_tokens)]).to(device)
    with torch.no_grad():
        encoded_layers, _ = cluster_model(token_tensor, mask_tensor)
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(1, len(indexed_tokens)):
        if tokenized_words[i] in word_list:
            words = get_knn(encoded_layers[0][i].cpu(), 700)
            words = filter_words(words, 8)
        else:
            words = []
        if len(words) >= 1:
            similar_char_dict[tokenized_words[i]] = words
        else:
            similar_char_dict[tokenized_words[i]] = [tokenized_words[i]]

    return similar_char_dict


if __name__ == '__main__':
    device = torch.device("cuda")
    cluster_model = BertModel.from_pretrained('bert-base-uncased')
    cluster_model.eval()
    cluster_model = cluster_model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    embedding_space = torch.load(args.embedding_data)
    word_list = joblib.load(args.word_list)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    test_data = YelpDataset(args.test_data)

