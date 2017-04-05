# -*- coding:utf8 -*-
import json
import codecs

dictionary = {}

file_in = json.load(open('./../data/train_data.json'), 'utf8')

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
with codecs.open("./../dict/dict_unsort.dict", 'w', 'utf8') as f_out:
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
with codecs.open("./../dict/dict_sorted.dict", 'w', 'utf8') as f_out:
    f_out.write('<unk>')
    f_out.write('\t')
    f_out.write(str(unk_count))
    f_out.write('\n')

    for word, freq in sorted_dict:
        f_out.write(word)
        f_out.write('\t')
        f_out.write(str(freq))
        f_out.write('\n')
'''

# 根据出现频率设置阈值,低于阈值的采用<unk>代替
threhold = 30000

unk_count = 0
for _, freq in sorted_dict[(threhold-1):]:
    unk_count += freq

dict_name = './../dict/dict_' + str(threhold) + '.dict'

with codecs.open(dict_name, 'w', 'utf8') as f_out:
    f_out.write('<unk>')
    f_out.write('\t')
    f_out.write(str(unk_count))
    f_out.write('\n')

    for word, freq in sorted_dict[:(threhold-1)]:
        f_out.write(word)
        f_out.write('\t')
        f_out.write(str(freq))
        f_out.write('\n')
