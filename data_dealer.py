import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import spacy
import string
import pickle
import yaml
import copy
from tqdm import tqdm


# get configuration from yaml file
def yaml_config():
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


# load data
def read_txt_file(file_path):
    with open(file_path, "r") as f:
        notes = [i.strip() for i in f.readlines()]
    return notes


class Tokenizer(object):
    def __init__(self, notes, dictionary_obj_path):
        self.vocab_notes = self.tokenize(notes)
        self.dictionary_obj = self.load_or_create_dictionary(self.vocab_notes, dictionary_obj_path)
        self.tokenized_ds = self.tokenize_dataset(self.vocab_notes, self.dictionary_obj)

    def token_prep(self, note):
        """
        input: one string of raw note for each record
        output: cleaned note ready for tokenization (w/o punctuation, extra white space)
        """
        note = re.sub(r'[^\w\s]', '', note)  # remove punctuation
        note = re.sub(' +', ' ', note)  # reduce to single white space
        return note

    def tokenize(self, notes):
        """
        input: list of raw notes for records
        output: list of lists of lower-case word tokens
        """
        notes = [self.token_prep(note) for note in notes]
        nlp = spacy.load("en_core_web_sm")
        tokenized_note = []
        for note in tqdm(notes):
            result = nlp(note) # tokenized document
            tokenized_note.append([token.text.lower() for token in result if token.text not in string.punctuation])
        return tokenized_note

    def load_or_create_dictionary(self, vocab_notes, dictionary_obj_path):
        if not os.path.exists(dictionary_obj_path):
            os.makedirs(dictionary_obj_path)

        full_file_path = os.path.join(dictionary_obj_path, 'dictionary.p')

        if os.path.isfile(full_file_path):
            dictionary_obj = pickle.load(open(full_file_path, "rb"))
        else:
            dictionary_obj = Dictionary(vocab_notes)
            pickle.dump(dictionary_obj, open(full_file_path, "wb"))
        return dictionary_obj

    def tokenize_dataset(self, vocab_notes, dictionary_obj):
        """
        input: vocab_notes: list of lists of lower-case word tokens
               dictionary: Dictionary object
        output: list of lists of indices with <s>
        """
        tokenized_ds = []
        for l in tqdm(vocab_notes):
            l = ['<s>'] + l + ['<s>'] # add <s>
            encoded_l = dictionary_obj.encode_token_seq(l)
            tokenized_ds.append(encoded_l)
        return tokenized_ds


# build the vocabulary
class Dictionary(object):
    def __init__(self, datasets):
        self.tokens = []
        self.ids = {}

        # add special tokens
        self.add_token('<s>')
        self.add_token('[M]')
        self.add_token('<pad>')
        self.add_token('<unk>')

        for line in tqdm(datasets):
            for w in line:
                self.add_token(w)

    def add_token(self, w):
        if w not in self.tokens:
            self.tokens.append(w)
            _w_id = len(self.tokens) - 1
            self.ids[w] = _w_id

    def get_id(self, w):
        return self.ids[w]

    def get_token(self, idx):
        return self.tokens[idx]

    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]

    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids['<unk>'] for i in l]

    def __len__(self):
        return len(self.tokens)


class SequenceDataset(Dataset):
    def __init__(self, list_of_token_lists):
        self.input_tensors = []
        for sample in list_of_token_lists:
            self.input_tensors.append(torch.tensor([sample], dtype=torch.long)) # put sample inside [] for unsqueeze

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx]


# padding functions
def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padded_tensor = torch.cat(
            [t, torch.tensor([[pad_token] * (max_length - t.size(-1))], dtype=torch.long)], dim=-1)
        padded_list.append(padded_tensor)

    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor

def pad_collate_fn(pad_idx, batch):
    input_list = [s for s in batch]
    input_tensor = pad_list_of_tensors(input_list, pad_idx)
    input_tensor = input_tensor.transpose(0, 1) # (length,batch,dim)
    return input_tensor


def mask_tokens(inputs, mask_prob, pad_token_id, mask_token_id, vsize):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    inputs = copy.deepcopy(inputs)
    labels = copy.deepcopy(inputs)
    # Sample masked tokens (mask_prob(%) for each batch)
    masked_indices = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool() # cast storage to bool type
    masked_indices = masked_indices & (inputs != pad_token_id)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked tokens with [M]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vsize, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
