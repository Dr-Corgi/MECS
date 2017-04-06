# -*- coding:utf8 -*-
import json
import codecs

def build_dict(file_name,
               dict_path,
               vocab_size=30000,
               encoding='utf8'):
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

    # 将词典中词语进行输出
    '''
    with codecs.open(dict_path, 'w', encoding) as f_out:
        for k in dictionary.keys():
            f_out.write(k)
            f_out.write('\t')
            f_out.write(str(dictionary[k]))
            f_out.write('\n')
    '''

    # 对词典中词语根据出现频率排序
    sorted_dict = sorted(dictionary.items(), lambda x,y: cmp(x[1], y[1]), reverse=True)

    # 将排序后的词典进行输出
    '''
    with codecs.open(dict_path, 'w', encoding) as f_out:
        f_out.write('<unk>\t')
        f_out.write(str(unk_count))
        f_out.write('\n')
        f_out.write('<pad>\t0\n')
        f_out.write('<start>\t0\n')
        f_out.write('<end>\t0\n')

        for word, freq in sorted_dict:
            f_out.write(word)
            f_out.write('\t')
            f_out.write(str(freq))
            f_out.write('\n')
    '''

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
        f_out.write('<start>\t0\n')
        f_out.write('<end>\t0\n')

        for word, freq in sorted_dict[:(vocab_size-4)]:
            f_out.write(word)
            f_out.write('\t')
            f_out.write(str(freq))
            f_out.write('\n')


def load_dict(dict_name,
              encoding='utf8'):
    word_to_idx = {}
    idx_to_word = {}

    with codecs.open(dict_name, 'r', encoding) as f_in:
        for line in f_in:
            word = line.strip().split("\t")[0]
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[len(idx_to_word)] = word

    return word_to_idx, idx_to_word

build_dict("./../data/train_data.json", './../dict/')