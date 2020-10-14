from metrics import Metrics
import time
import data_utils
import pickle


from model.bilstmcrf import BILSTM_Model
from model_utils import result_to_json


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, args, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    with open(args.map_path, 'wb') as f:
        pickle.dump([word2id, tag2id], f)

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    data_utils.save_config(args.config_path, vocab_size, out_size)

    bilstm_model = BILSTM_Model(vocab_size, out_size, args)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf"
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists, test_tag_lists


def predict_line(args):
    while True:
        try:
            input_line = input('请输入要预测的句子(退出请输入q):')
            if input_line == 'q':
                break
            with open(args.map_path, 'rb') as f:
                word2id, tag2id = pickle.load(f)
            word_list = data_utils.input_from_line(input_line)
            config = data_utils.load_config(args.config_path)
            vocab_size, out_size = config['vocab_size'], config['out_size']
            model = BILSTM_Model(vocab_size, out_size, args)

            pred = model.predict(word_list, word2id, tag2id)[0]
            result = result_to_json(input_line, pred)
            print(result)
        except IndexError:
            continue


