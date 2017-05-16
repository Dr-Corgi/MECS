from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import platform

from tensorflow.python.platform import gfile

from tf_chatbot.configs.config import BUCKETS


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

_ENCODING = "utf8"


def DICT_LIST(dic):
    return {k: [] for k in dic.keys()}


def get_dialog_train_set_path(path):
    return os.path.join(path, 'train_data')


def get_dialog_dev_set_path(path):
    return os.path.join(path, 'dev_data')


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                            tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        data = json.load(open(data_path))
        counter = 0
        for ((q,qe),(a,ae)) in data:
            counter += 1
            if counter % 50000 == 0:
                print("  Create_vocabulary: processing line %d" % counter)

            tokens_q = tokenizer(q) if tokenizer else basic_tokenizer(q)
            for tok in tokens_q:
                word = re.sub(_DIGIT_RE, "0", tok) if normalize_digits else tok
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

            tokens_a = tokenizer(a) if tokenizer else basic_tokenizer(a)
            for tok in tokens_a:
                word = re.sub(_DIGIT_RE, "0", tok) if normalize_digits else tok
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode='w') as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + '\n')


def initialize_vocabulary(vocabulary_path):

    if gfile.Exists(vocabulary_path):
        rev_vocab = []

        with gfile.GFile(vocabulary_path, mode='r') as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])     # {'word':index}
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found" % vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)

    if platform.system() == "Windows":
        if not normalize_digits:
            return [vocabulary.get(w, UNK_ID) for w in words]
        return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]
    else:
        if not normalize_digits:
            return [vocabulary.get(w.encode('utf8'), UNK_ID) for w in words]
        return [vocabulary.get(re.sub(_DIGIT_RE, "0", w.encode('utf8')), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(target_path, mode='w') as tokens_file:
            data = json.load(open(data_path))
            counter = 0
            for ((q,qe),(a_0,ae_0),(a_1,ae_1),(a_2,ae_2),(a_3,ae_3),(a_4,ae_4),(a_5,ae_5)) in data:
                counter += 1
                if counter % 50000 == 0:
                    print("  Data_to_token_ids: tokenizing line %d" % counter)
                token_ids_q = sentence_to_token_ids(q, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_q]) + '\n')
                token_ids_a0 = sentence_to_token_ids(a_0, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a0]) + '\n')
                token_ids_a1 = sentence_to_token_ids(a_1, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a1]) + '\n')
                token_ids_a2 = sentence_to_token_ids(a_2, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a2]) + '\n')
                token_ids_a3 = sentence_to_token_ids(a_3, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a3]) + '\n')
                token_ids_a4 = sentence_to_token_ids(a_4, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a4]) + '\n')
                token_ids_a5 = sentence_to_token_ids(a_5, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids_a5]) + '\n')


def prepare_dialog_data(data_dir, vocabulary_size):
    train_path = get_dialog_train_set_path(data_dir)
    dev_path = get_dialog_dev_set_path(data_dir)

    vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
    create_vocabulary(vocab_path, train_path+"_origin.json", vocabulary_size)

    train_ids_path = train_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(train_path + ".json", train_ids_path, vocab_path)

    dev_ids_path = dev_path + (".ids%d.in" % vocabulary_size)
    data_to_token_ids(dev_path + ".json", dev_ids_path, vocab_path)

    return (train_ids_path, dev_ids_path, vocab_path)


def read_data(tokenized_dialog_path, max_size=None):

    data_set = [[] for _ in BUCKETS]

    with gfile.GFile(tokenized_dialog_path, mode='r') as fh:
        counter = 0
        source, emo0,emo1,emo2,emo3,emo4,emo5 = fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline()
        while source and emo0 and emo1 and emo2 and emo3 and emo4 and emo5 and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)

            source_ids = [int(x) for x in source.split()]
            emo0_ids = [int(x) for x in emo0.split(" ")]+[EOS_ID]
            emo1_ids = [int(x) for x in emo1.split(" ")]+[EOS_ID]
            emo2_ids = [int(x) for x in emo2.split(" ")]+[EOS_ID]
            emo3_ids = [int(x) for x in emo3.split(" ")]+[EOS_ID]
            emo4_ids = [int(x) for x in emo4.split(" ")]+[EOS_ID]
            emo5_ids = [int(x) for x in emo5.split(" ")]+[EOS_ID]

            target = {0:emo0_ids, 1:emo1_ids, 2:emo2_ids,
                      3:emo3_ids, 4:emo4_ids, 5:emo5_ids}

            target_maxlen = max([len(ids) for _,ids in target.items()])

            for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
                if len(source_ids) < source_size and target_maxlen < target_size:
                    data_set[bucket_id].append([source_ids, target])
                    break
            source, emo0,emo1,emo2,emo3,emo4,emo5 = fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline(),fh.readline()

    return data_set