import torch
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from ast import literal_eval
from data_dealer import read_txt_file, Tokenizer


def load_attn_pickle(file_path):
    with open (file_path, 'rb') as fp:
        attention = pickle.load(fp)
    return attention


def load_vocab_notes(file_path, vocab_obj_path):
    notes = read_txt_file(file_path)
    tokenizer_obj = Tokenizer(notes, vocab_obj_path)
    vocab_notes = tokenizer_obj.vocab_notes
    return vocab_notes


def load_label(file_path, upper=55000, lower=None):
    df = pd.read_csv(file_path, converters={'token_head_pair':literal_eval})
    pair_holder = {}
    for i, pair_list in enumerate(df.token_head_pair[upper:lower]):
        pair_holder[i] = pair_list
    return pair_holder


def save_json(data, file_path):
    with open(file_path, 'w') as fp:
        json.dump(data, fp)

# with open('data.json', 'r') as fp:
#     data = json.load(fp)


def reshape_attention(layer_attn):
    """
    Input:
        layer_attn: list of n_layer elements with shape of (batch_size, n_head, len_q, len_k)
    Returns:
        Tensor of (batch_size, n_layer, n_head, len_q, len_k)
    """
    len_qk = layer_attn[0].shape[-1]
    reshape_attn = torch.cat(layer_attn, dim=0).reshape(2, 8, 8, len_qk, len_qk)
    # (n_layer, batch_size, n_head, len_q, len_k)

    reshape_attn = reshape_attn.transpose(0, 1)  # (batch_size, n_layer, n_head, len_q, len_k)
    return reshape_attn


def most_attended_to_word(qk_attn, token_list):
    """
    Input:
        qk_attn: Tensor of (len_q, len_k)
        token_list: list of tokens from one record without "<s>"
    Returns:
        list of tuples of (query_token, most_attended_to_word)
    """
    most_attended_to_idx = torch.argmax(qk_attn, dim=-1)[1:-1]

    pair_holder = []
    token_list_ = ['<s>'] + token_list + ['<s>']
    for token, idx in zip(token_list, most_attended_to_idx):
        if idx > (len(token_list) + 1):
            most_attended_to_word = "<pad>"
        else:
            most_attended_to_word = token_list_[idx]
        pair_holder.append((token, most_attended_to_word))

    return pair_holder


def single_record_result(layer_head_attn, token_list):
    """
    Input:
        layer_head_attn: Tensor of (n_layer, n_head, len_q, len_k)
        token_list: list of tokens from one record without "<s>"
    Returns:
        dictionary of key of "head<layer>-<index>" and value of [(query_token, most_attended_to_word), ...]
    """
    n_layer = layer_head_attn.shape[0]
    n_head = layer_head_attn.shape[1]
    pair_dict = {}

    for i_layer in range(n_layer):
        for i_head in range(n_head):
            attn = layer_head_attn[i_layer][i_head]
            pair_dict["head{}-{}".format(i_layer, i_head)] = most_attended_to_word(attn, token_list)
    return pair_dict


def all_record_result(attn, token_notes, json_path=None):
    """
    Input:
        attn: list of n_iteration elements,
              each element is a list of n_layer elements with shape of (batch_size, n_head, len_q, len_k)
        token_notes: list of tokenized notes without "<s>"
    Returns:
        dictionary of key of <record_index> and value of dictionary of each layer-head combination result
    """
    batch_size = attn[0][0].shape[0]
    result_dict = {}

    for i_iter, layer_attn in enumerate(attn):
        reshape_attn = reshape_attention(layer_attn)  # (batch_size, n_layer, n_head, len_q, len_k)
        for i_record in range(batch_size):
            i_note = i_iter * batch_size + i_record
            pair_dict = single_record_result(reshape_attn[i_record], token_notes[i_note])
            result_dict[i_note] = pair_dict

    if json_path is not None:
        save_json(result_dict, json_path)

    return result_dict


def compare_tuple_list(layer_head_list, label_list):
    """
    Input:
        layer_head_list: list of tuples of (query_token, most_attended_to_word) for a layer-head combination
        label_list: list of tuples of (query_token, token_head) as label
    Returns:
        correct_match (int): number of query tokens whose most-attended-to word matches its syntactic head
    """
    correct_match = 0
    for pair, pair_label in zip(layer_head_list, label_list):
        if pair[0] != pair_label[0].lower():
            raise ValueError("The query tokens do not match.")
        else:
            if pair[1] == pair_label[1].lower():
                correct_match += 1
            else:
                pass
    return correct_match


def single_record_counter(layer_head_dict, label_list, counter_dict, len_list, record_idx):
    """
    Input:
        layer_head_dict: dictionary of key of "head<layer>-<index>"
                         and value of [(query_token, most_attended_to_word), ...]
        label_list: list of tuples of (query_token, token_head) as label
        counter_dict: dictionary of key of "head<layer>-<index>"
                      and value of list of number of query tokens whose most-attended-to word matches its syntactic head
        len_list: list of total number of query tokens for each record
        record_idx: index of the record
    Returns:
        updated counter_dict and len_list with numbers added for one record
    """
    for layer_head in list(layer_head_dict.keys()):
        layer_head_list = layer_head_dict[layer_head]

        if len(layer_head_list) != len(label_list):
            print("For record {}: Two lists should have the same number of element tuples.".format(record_idx + 1))
            break
        else:
            correct_match = compare_tuple_list(layer_head_list, label_list)
            counter_dict[layer_head].append(correct_match)
            if layer_head == "head0-0":
                len_list.append(len(label_list))
    return counter_dict, len_list


def all_record_counter(result_dict, label_dict):
    """
    Input:
        result_dict: dictionary of key of <record_index>
                     and value of dictionary of each layer-head combination result
        label_dict: dictionary of key of <record_index>
                    and value of list of tuples of (query_token, token_head) as label
    Returns:
        counter_dict: dictionary of key of "head<layer>-<index>"
                      and value of list of number of query tokens whose most-attended-to word matches its syntactic head
        len_list: list of total number of query tokens for each record
    """
    counter_dict = defaultdict(list)
    len_list = []

    for record_idx in list(result_dict.keys()):
        layer_head_dict = result_dict[record_idx]
        label_list = label_dict[record_idx]
        counter_dict, len_list = single_record_counter(layer_head_dict, label_list,
                                                       counter_dict, len_list, record_idx)
    return counter_dict, len_list


def calculate_precision(counter_dict, len_list):
    """
    Input:
        counter_dict: dictionary of key of "head<layer>-<index>"
                      and value of list of number of query tokens whose most-attended-to word matches its syntactic head
        len_list: list of total number of query tokens for each record
    Returns:
        precision_dict: dictionary of key of "head<layer>-<index>"
                        and value of precision for each layer-head combination
    """
    precision_dict = defaultdict(list)
    total_tokens = np.sum(len_list)
    for layer_head in list(counter_dict.keys()):
        precision_dict[layer_head] = np.sum(counter_dict[layer_head]) / total_tokens
    return precision_dict

