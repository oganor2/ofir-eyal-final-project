import torch
import numpy as np
import os
import pandas as pd
import transformers
import re
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
from transformers import logging as t_log
from sklearn.metrics import f1_score
import logging
import copy
from tqdm import tqdm
import warnings
t_log.set_verbosity_error()
warnings.filterwarnings("ignore")
tqdm.pandas()
torch.random.manual_seed(42)
device = torch.cuda.is_available()
punc = ['!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'None']
annotations_embedding = {punc[i]: i for i in range(len(punc))}
reverse_annotations_embedding = {val: key for key, val in annotations_embedding.items()}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class SentenceDataSet(torch.utils.data.Dataset):
    def __init__(self, data, max_len):
        self.sentences = data['sentences']
        self.labels = data['labels']
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sentences[idx]['input_ids'])
        attention_mask = torch.tensor(self.sentences[idx]['attention_mask'])
        # fill empty spaces (padding) with -1
        labels = torch.tensor(self.labels[idx]
                              + [[-1] * 4 for i in range(len(input_ids) - len(self.labels[idx]))])
        before, after, capital, br = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]
        return input_ids[:self.max_len], attention_mask[:self.max_len], before[:self.max_len], \
               after[:self.max_len], capital[:self.max_len], br[:self.max_len]


def find_nth(haystack, needle, n):
    """
    simple utility for finding the n'th appearance of a term
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def clean_text(text, n=50):
    """
    removes all undesired characters, and concatenates paragraph breaks for simplicity
    returns the text except for first n paragraphs (might include intro, copyrights etc.)
    """
    for i in reversed(range(2, 10)):
        text = text.replace('\n' * i, ' <br> ').replace('-' * i, '')
    text = text.replace('\n', ' ').replace('\t', ' ').replace('[', '(').replace(']', ')')
    text = re.sub('([_*|])', '', text)
    # add spacing around chars
    text = re.sub('([.,!?():"\';,-])', r' \1 ', text)
    # remove double spaces
    text = re.sub('\s{2,}', ' ', text)
    # remove first n paragraphs since they tend to contain meta text
    if n:
        text = text[find_nth(text, '<br>', n) + 4:]
    return text


def annotate_text(text, raw=False):

    text_list = text.split(' ')
    annotations = []
    length = len(text_list)
    punc = ['.', ',', '!', '?', '(', ')', ':', '"', '\'', ';', '-']
    for i in range(length):
        annotation = {'before': '', 'after': '', 'capital': '', 'break': ''}
        if raw:
            annotation = {'before': -1, 'after': -1, 'capital': -1, 'break': -1}
        else:
            if text_list[i] in ['.', ',', '!', '?', '(', ')', ':', '"', '\'', ';', '-', '<br>'] or not \
            text_list[i]:
                continue
            if i and text_list[i - 1] in punc:
                annotation['before'] = text_list[i - 1]
            else:
                annotation['before'] = 'None'
            if i < length - 1 and text_list[i + 1] in punc:
                annotation['after'] = text_list[i + 1]
            else:
                annotation['after'] = 'None'
            if text_list[i].lower() != text_list[i]:
                annotation['capital'] = 1
            else:
                annotation['capital'] = 0
            if (i < length - 1 and text_list[i + 1] == '<br>') or (
                    i < length - 2 and text_list[i + 1] in punc and text_list[i + 2] == '<br>'):
                annotation['break'] = 1
            else:
                annotation['break'] = 0
        annotations.append(annotation)
    return annotations


def to_raw(text):
    return [i for i in re.sub('([.,!?():"\';,-])', '', text).replace('<br>', '').lower().split(" ") if i]


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, seq_len):
    """
    Solves the issue with bert tokenizer breaking up words into partial words, adjusts labels accordingly
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    tokenized_sentence += [0 for i in range(seq_len - len(tokenized_sentence))]
    mask = [1 if i else 0 for i in tokenized_sentence]
    return {'input_ids': tokenized_sentence, 'attention_mask': mask}, labels


def prepare_for_batches(text, annotations, tokenizer, actual_seq_len, label_to_idx,
                        val_percent=0.15, test_percent=0.1):
    """
    takes words, annotations and creates sequences for the model
    :param text: the raw text (after pipeline)
    :param annotations: annotations for the text
    :param tokenizer: the tokenizer to use for embedding
    :param actual_seq_len: desired sequence len for the model
    :param label_to_idx: conversion of labels (char) to int
    :param val_percent: percent to use for validation
    :param test_percent: percent to use for test
    :return: dictionary of train,val,test and their data (can be empty)
    """
    seq_len = int(0.7 * actual_seq_len)
    # create sequences
    sentences = [text[i:i + seq_len] for i in range(0, len(text), seq_len)]
    labels = [annotations[i:i + seq_len] for i in range(0, len(annotations), seq_len)]
    # adjust to tokenizer
    adjusted = [tokenize_and_preserve_labels(sentences[i], labels[i], tokenizer, actual_seq_len)
                for i in range(len(sentences))]
    tokenized_sentences = [i[0] for i in adjusted]
    tokenized_labels = [i[1] for i in adjusted]
    # convert annotations to numerical labels
    tokenized_labels = [[[label_to_idx[i[l]] if type(i[l]) == str else i[l] for l in i] for i in seq]
                        for seq in tokenized_labels]
    # create train, val, test
    test_index = int(len(tokenized_sentences) * (1 - test_percent))
    val_index = test_index - int(len(tokenized_sentences) * val_percent)
    train = (tokenized_sentences[:val_index], tokenized_labels[:val_index])
    val = (tokenized_sentences[val_index:test_index], tokenized_labels[val_index:test_index])
    test = (tokenized_sentences[test_index:], tokenized_labels[test_index:])
    return {'train': train, 'val': val, 'test': test}


def data_to_dataset(data, seq_len, train_val=True):
    """
    takes output of cleaning pipeline divided into sequences and converts to a torch dataset
    :param data: text and annotations from pipeline
    :param seq_len: desired sequence length
    :param train_val: whether train/evaluate is running or inference
    :return: torch dataset(s)
    """
    train, val, test = {'sentences': [], 'labels': []}, {'sentences': [], 'labels': []}, \
                       {'sentences': [], 'labels': []}
    for i in data:
        if train_val:
            train['sentences'].extend(i['train'][0]), train['labels'].extend(i['train'][1])
            val['sentences'].extend(i['val'][0]), val['labels'].extend(i['val'][1])
        test['sentences'].extend(i['test'][0]), test['labels'].extend(i['test'][1])
    if train_val:
        train, val, test = SentenceDataSet(train, seq_len), SentenceDataSet(val,
                                                                            seq_len), SentenceDataSet(
            test, seq_len)
        return train, val, test
    else:
        return SentenceDataSet(test, seq_len)


def translate_annotation(annotation):
    translated = {}
    translated['before'] = reverse_annotations_embedding[annotation['before']]
    translated['after'] = reverse_annotations_embedding[annotation['after']]
    translated['capital'] = annotation['capital']
    translated['break'] = annotation['break']
    return translated


def get_optimal_annotation(pred_1, pred_2):
    """
    return the best annotation- if after was predicted as nothing and before as something,
    return the one that says something.
    :param pred_1: prediction for after of i-1
    :param pred_2: prediction for before of i
    :return: best prediction
    """
    annotations = []
    ignore_values = ['None']
    comp_w_others = ['(', ')','"',"",]
    if pred_1 in ignore_values:
        pred_1 = ""
    if pred_2 in ignore_values:
        pred_2 = ""
    # if same, return one of them. Note that identical to last case, separated to simplify logic
    if pred_1 == pred_2:
        annotations.extend([pred_1,""])
    # if possible to have two different ones, return both
    elif pred_1 in comp_w_others or pred_2 in comp_w_others:
        if pred_1:
            annotations.extend([pred_1,pred_2])
        else:
            annotations.extend([pred_2,pred_1])
    # if conflicting annotations and neither are compatible with others, the prediction for
    # after is better empirically, so we'll stick with it
    else:
        annotations.extend([pred_1,""])
    return annotations
