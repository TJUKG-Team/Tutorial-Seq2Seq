from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe
import spacy
import torch
import dill
import os


class Multi30k_Dataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        return data.interleave_keys(len(example.src), len(example.trg))


class Multi30k():
    def __init__(self, batch_size, device):
        print('Load Multi30k ...')
        # config path
        self.save_path = f'datasets/processed/multi30k'
        self.save_dataset_path = {
            'train_dataset': f'{self.save_path}/train_dataset',
            'vaild_dataset': f'{self.save_path}/vaild_dataset',
            'test_dataset': f'{self.save_path}/test_dataset'
        }
        self.save_vocab_path = {
            'src': f'{self.save_path}/src_vocab',
            'trg': f'{self.save_path}/trg_vocab'
        }

        # 定义Field
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.SRC = data.Field(tokenize=self.tokenize_src, init_token='<sos>', eos_token='<eos>', lower=True)
        self.TRG = data.Field(tokenize=self.tokenize_trg, init_token='<sos>', eos_token='<eos>', lower=True)

        # 构建dataset
        if self.if_split_already():  # data.TabularDataset能够通过dataset.examples和fields重构出来
            fields = {'src': self.SRC, 'trg': self.TRG}
            self.train_dataset, self.valid_dataset, self.test_dataset = self.load_split_datasets(fields)
        else:
            self.train_dataset, self.valid_dataset, self.test_dataset = datasets.Multi30k.splits(root='datasets',
                                                                                                 exts=('.de', '.en'),
                                                                                                 fields=(self.SRC, self.TRG))
            self.dump_split_datasets(self.train_dataset, self.valid_dataset, self.test_dataset)

        # 构建vocab
        if self.if_vocab_already():
            self.SRC.vocab, self.TRG.vocab = self.load_vocab()
        else:
            self.SRC.build_vocab(self.train_dataset, min_freq=2, vectors=GloVe(name='840B', dim=300,
                                                                               cache='datasets/.vector_cache'))
            self.TRG.build_vocab(self.train_dataset, min_freq=2, vectors=GloVe(name='840B', dim=300,
                                                                               cache='datasets/.vector_cache'))
            self.dump_vocab(self.SRC.vocab, self.TRG.vocab)

        # 构建iterator
        self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits((self.train_dataset,
                                                                                                   self.valid_dataset,
                                                                                                   self.test_dataset),
                                                                                                  batch_size=batch_size,
                                                                                                  device=device)

    def tokenize_src(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)][::-1]

    def tokenize_trg(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def if_split_already(self):
        for dataset_path in self.save_dataset_path.values():
            if not os.path.exists(dataset_path):
                return False
        return True

    def dump_split_datasets(self, train_dataset, vaild_dataset, test_dataset):
        os.makedirs(self.save_path, exist_ok=True)
        with open(self.save_dataset_path['train_dataset'], 'wb') as f:
            dill.dump(train_dataset.examples, f)
        with open(self.save_dataset_path['vaild_dataset'], 'wb') as f:
            dill.dump(vaild_dataset.examples, f)
        with open(self.save_dataset_path['test_dataset'], 'wb') as f:
            dill.dump(test_dataset.examples, f)

    def load_split_datasets(self, fields):
        with open(self.save_dataset_path['train_dataset'], 'rb') as f:
            train_examples = dill.load(f)
        with open(self.save_dataset_path['vaild_dataset'], 'rb') as f:
            vaild_examples = dill.load(f)
        with open(self.save_dataset_path['test_dataset'], 'rb') as f:
            test_examples = dill.load(f)
        train_dataset = Multi30k_Dataset(examples=train_examples, fields=fields)
        vaild_dataset = Multi30k_Dataset(examples=vaild_examples, fields=fields)
        test_dataset = Multi30k_Dataset(examples=test_examples, fields=fields)
        return train_dataset, vaild_dataset, test_dataset

    def if_vocab_already(self):
        for vocab_path in self.save_vocab_path.values():
            if not os.path.exists(vocab_path):
                return False
        return True

    def dump_vocab(self, src_vocab, trg_vocab):
        with open(self.save_vocab_path['src'], 'wb') as f:
            dill.dump(src_vocab, f)
        with open(self.save_vocab_path['trg'], 'wb') as f:
            dill.dump(trg_vocab, f)

    def load_vocab(self):
        with open(self.save_vocab_path['src'], 'rb') as f:
            src_vocab = dill.load(f)
        with open(self.save_vocab_path['trg'], 'rb') as f:
            trg_vocab = dill.load(f)
        return src_vocab, trg_vocab

    def get_data_iterator(self):
        return self.train_iterator, self.valid_iterator, self.test_iterator

    def get_Fields(self):
        return self.SRC, self.TRG

    def get_sos_eos_indexes(self):
        tokens_indexes = [[self.TRG.vocab.stoi[self.TRG.init_token]], [self.TRG.vocab.stoi[self.TRG.eos_token]]]
        return torch.tensor(tokens_indexes)

    def encode_src_sentence(self, src):
        src_tokens = self.tokenize_src(src)
        src_tokens = [self.SRC.eos_token] + src_tokens + [self.SRC.init_token]
        src_indexes = [self.SRC.vocab.stoi[token.lower()] for token in src_tokens]
        return torch.tensor(src_indexes).unsqueeze(1)

    def decode_trg_sentence(self, trg_indexes):
        trg_tokens = [self.TRG.vocab.itos[index] for index in trg_indexes]
        return ' '.join(trg_tokens)
