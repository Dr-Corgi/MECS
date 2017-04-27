# -*- coding:utf8 -*-
import json
import codecs
from conf.profile import TOKEN_UNK, TOKEN_EOS, TOKEN_BOS, TOKEN_PAD
import numpy as np

def build_dict(file_name, dict_path, vocab_size=30000, encoding='utf8'):

    dictionary = {}

    file_in = json.load(open(file_name), encoding)

    for line in file_in:
        question, answer = line
        q_text, q_label = question
        a_text, a_label = answer

        q_words = q_text.strip().split(" ")
        for word in q_words:
            dictionary[word] = dictionary.get(word, 0) + 1

        a_words = a_text.strip().split(" ")
        for word in a_words:
            dictionary[word] = dictionary.get(word, 0) + 1

    # 对词典中词语根据出现频率排序
    sorted_dict = sorted(dictionary.items(), lambda x,y: cmp(x[1], y[1]), reverse=True)

    # 根据出现频率设置阈值,低于阈值的采用<unk>代替
    unk_count = 0
    for _, freq in sorted_dict[(vocab_size-4):]:
        unk_count += freq

    dict_name = dict_path+'dict_' + str(vocab_size) + '.dict'

    with codecs.open(dict_name, 'w', encoding) as f_out:
        f_out.write('<unk>\t')
        f_out.write(str(unk_count))
        f_out.write('\n')
        f_out.write('<pad>\t0\n')
        f_out.write('<bos>\t0\n')
        f_out.write('<eos>\t0\n')

        for word, freq in sorted_dict[:(vocab_size-4)]:
            f_out.write(word)
            f_out.write('\t')
            f_out.write(str(freq))
            f_out.write('\n')


def load_dict(dict_name, encoding='utf8'):

    word_to_idx = {}
    idx_to_word = {}

    with codecs.open(dict_name, 'r', encoding) as f_in:
        for line in f_in:
            word = line.strip().split("\t")[0]
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[len(idx_to_word)] = word

    return word_to_idx, idx_to_word

def loadPretrainedVector(vocab_size, embedding_size, pretrain_file):
    vocab_to_idx = {}
    idx_to_vocab = {}
    vocab_embed = []
    with open(pretrain_file, encoding='utf8') as fin:
        line_0 = fin.readline().strip().split(" ")
        assert int(line_0[0]) >= vocab_size
        assert int(line_0[1]) == embedding_size

        # pad
        #vec = [0.0 for i in range(embedding_size)]
        vec = list(np.random.uniform(10.0, -10.0, [embedding_size]))
        vocab_to_idx[TOKEN_PAD] = len(vocab_to_idx)
        idx_to_vocab[len(idx_to_vocab)] = TOKEN_PAD
        vocab_embed.append(vec)

        # start & end
        vec = list(np.random.uniform(10.0, -10.0, [embedding_size]))
        vocab_to_idx[TOKEN_BOS] = len(vocab_to_idx)
        idx_to_vocab[len(idx_to_vocab)] = TOKEN_BOS
        vocab_embed.append(vec)

        vec = list(np.random.uniform(10.0, -10.0, [embedding_size]))
        vocab_to_idx[TOKEN_EOS] = len(vocab_to_idx)
        idx_to_vocab[len(idx_to_vocab)] = TOKEN_EOS
        vocab_embed.append(vec)

        # unk
        vec = list(np.random.uniform(10.0, -10.0, [embedding_size]))
        #vec = list(2 * np.mean(vocab_embed, axis=0))
        vocab_to_idx[TOKEN_UNK] = len(vocab_to_idx)
        idx_to_vocab[len(idx_to_vocab)] = TOKEN_UNK
        vocab_embed.append(vec)


        # words
        for i in range(vocab_size - 4):
            line_ = fin.readline().strip().split(" ")

            voc = line_[0]
            #print(type(voc))
            vec = [float(n) for n in line_[1:]]
            vocab_to_idx[voc] = len(vocab_to_idx)
            idx_to_vocab[len(idx_to_vocab)] = voc
            vocab_embed.append(vec)

    return vocab_to_idx, idx_to_vocab, np.asarray(vocab_embed)