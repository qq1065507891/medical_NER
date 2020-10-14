#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author JackZeng
import os
import codecs


label_dict = {
                      '检查和检验': 'CHECK',
                      '症状和体征': 'SIGNS',
                      '疾病和诊断': 'DISEASE',
                      '治疗': 'TREATMENT',
                      '身体部位': 'BODY'}


def get_data_path(path):
    """
    获取数据集的地址
    :param path:
    :return:
    """
    paths = []
    data_labels_paths = []
    for dir in os.listdir(path):
        data_paths = os.path.join(path, dir)
        for data_path in os.listdir(data_paths):
            if 'txtoriginal.txt' in os.path.join(data_paths,  data_path):
                paths.append(os.path.join(data_paths,  data_path))
            else:
                data_labels_paths.append(os.path.join(data_paths,  data_path))
    return paths, data_labels_paths


def transfrom_bioes(paths, label_paths, transfrom_path):
    """
    :param transfrom_path:
    :param paths:
    :param label_paths:
    :return:
    """
    f = codecs.open(transfrom_path, 'w+', encoding='utf-8')

    for data_path, label_path in zip(paths, label_paths):
        with codecs.open(data_path, 'r', encoding='utf-8') as f_data:
            content = f_data.read().strip()
            print('content', content)
            res_dict = {}
            with codecs.open(label_path, 'r', encoding='utf-8') as f_label:
                for label_line in f_label.readlines():
                    label_line = label_line.strip().split('\t')
                    start = int(label_line[1])
                    end = int(label_line[2])
                    label = label_line[3]
                    label_id = label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start and start == end:
                            label_cate = 'S-' + label_id
                        elif i == start and start != end:
                            label_cate = 'B-' + label_id
                        elif i == end:
                            label_cate = 'E-' + label_id
                        else:
                            label_cate = 'I-' + label_id
                        res_dict[i] = label_cate

                    for i, char in enumerate(content):
                        char_label = res_dict.get(i, 'O')
                        f.write(char + '\t' + char_label + '\n')
                    f.write(' \n')
        f.flush()
    return


def get_data_dic(paths):
    """
    获取特定切词字典
    :param path:
    :return:
    """
    labels = []
    f = codecs.open('data/dict_txt', 'w+', encoding='utf-8')
    for path in paths:
        with codecs.open(path, 'r', encoding='utf-8') as f_label:
            for label_line in f_label.readlines():
                label_line = label_line.strip().split('\t')[0]
                if label_line not in labels:
                    labels.append(label_line)
    for label in labels:
        f.write(label + '\n')
    f.close()


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data_origin')
    data_path, label_path = get_data_path(path)
    get_data_dic(label_path)
    # train_data_path, train_label_path = data_path[:720], label_path[:720]
    # dev_data_path, dev_label_path = data_path[720: 1079], label_path[720:1079]
    # test_data_path, test_label_path = data_path[1079:], label_path[1079:]
    # transfrom_bioes(train_data_path, train_label_path, 'data/train.txt')
    # transfrom_bioes(dev_data_path, dev_label_path, 'data/dev.txt')
    # transfrom_bioes(test_data_path, test_label_path, 'data/text.txt')
