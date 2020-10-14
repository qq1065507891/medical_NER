#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author JackZeng

import argparse
import torch
import numpy as np

import data_utils
import train


parse = argparse.ArgumentParser(description='Medical_NER')

parse.add_argument('--train', type=bool, default=False, help='是否训练')

parse.add_argument('--word_dim', type=int, default=100, help='word_dim')
parse.add_argument('--lstm_dim', type=int, default=100, help='lstm_dim')
parse.add_argument('--num_layers', type=int, default=2, help='lstm层数')

parse.add_argument('--batch_size', type=int, default=30, help='batch_size')
parse.add_argument('--lr', type=float, default=0.000001, help='learning_rate')
parse.add_argument('--require_improvement', type=int, default=200, help='多久模型没有提升')


parse.add_argument('--save_model', type=str, default='./save_model', help='模型保存路径')
parse.add_argument('--train_path', type=str, default='./data/train', help='训练数据路径')
parse.add_argument("--test_path", type=str, default='./data/test', help='测试数据路径')
parse.add_argument('--dev_path', type=str, default='./data/dev', help='验证数据路劲')
parse.add_argument('--config_path', type=str, default='./config.json', help='配置文件保存路径')
parse.add_argument('--map_path', type=str, default='./map.pkl', help='数据文件')

parse.add_argument('--max_epoch', type=int, default=100, help='最大轮训次数')
parse.add_argument('--step_check', type=int, default=5, help='step pre check')

args = parse.parse_args()


def main_train():
    print('开始加载数据')
    # 加载数据集
    train_word_lists, train_tag_lists, word2id, tag2id = \
        data_utils.my_build_corpus(args.train_path)
    dev_word_lists, dev_tag_lists = data_utils.my_build_corpus(args.dev_path, make_vocab=False)
    test_word_lists, test_tag_lists = data_utils.my_build_corpus(args.test_path, make_vocab=False)

    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    crf_word2id, crf_tag2id = data_utils.extend_maps(word2id, tag2id)
    # 还需要额外的一些数据处理d
    train_word_lists, train_tag_lists = data_utils.prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = data_utils.prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = data_utils.prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )

    lstmcrf_pred = train.bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id, args
    )
    print('over')


def run():
    if args.train:
        main_train()
    else:
       train.predict_line(args)


if __name__ == '__main__':
    run()

