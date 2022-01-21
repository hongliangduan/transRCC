#!/python

# version: 0.0.1

import json
from os import times
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from DeepZone.models import build_model_fn, build_kwargs
from DeepZone.models.Transformer.transformer import Transformer
from utils import *

parser = argparse.ArgumentParser(description='evaluate Transformer')

parser.add_argument('--data_dir',
                    action='store',
                    type=str,
                    default='data',
                    help='data directory.')

parser.add_argument('--out_dir',
                    action='store',
                    type=str,
                    default='out',
                    help='data output directory.')

parser.add_argument('--batch_size',
                    action='store',
                    type=int,
                    default=32,
                    help='batch_size')

parser.add_argument('--model_dir',
                    action='store',
                    type=str,
                    default='model',
                    help='where is the trained model')

parser.add_argument('--model_name',
                    action='store',
                    type=str,
                    default='ChemTrm',
                    help='project name for this training')

parser.add_argument('--dict_',
                    action='store',
                    type=str,
                    default='word2idx.json',
                    help='vocabulary of the dataset')


def get_smiles(sent_list, vocab_set, sign_start, sign_end, idx2word_vocab):
    smiles = ''
    for char in sent_list:
        if char == idx2word_vocab[vocab_set[sign_end]]:
            break
        elif char != idx2word_vocab[vocab_set[sign_start]]:
            smiles += char
        else:
            pass
    return smiles


if __name__ == '__main__':
    args = parser.parse_args()
    # get arguments
    data_dir = args.data_dir
    out_dir = args.out_dir
    batch_size = args.batch_size
    model_dir = args.model_dir
    model_name = args.model_name
    dict_ = args.dict_

    # read data from files
    sign_pad = '<pad>'
    data, vocab, sign_start, sign_end = read_data(data_dir)
    test_data = data['test']

    # load word2idx
    with open(dict_, 'r') as ifile:
        vocab_set = json.load(ifile)
    print('vocab loaded.')

    idx2word_vocab = inverse_dict(vocab_set)
    vocab_size = len(vocab_set)
    print('vocabulary size={}'.format(vocab_size))
    test_data = trans_data(test_data, vocab_set)

    # setup parameters
    n_heads = 8
    emb_dim = 256
    num_layers = 6
    FFN_inner_units = 2048
    dropout_keep_prob = 0.7

    kwargs = build_kwargs(voca_size=vocab_size,
                          label_size=vocab_size,
                          code_of_start=vocab_set[sign_start],
                          code_of_end=vocab_set[sign_end],
                          code_of_pad=vocab_set[sign_pad],
                          num_layers_enc=num_layers,
                          num_layers_dec=num_layers,
                          emb_dim=emb_dim,
                          n_heads=n_heads,
                          dropout_keep_prob=dropout_keep_prob,
                          FFN_inner_units=FFN_inner_units)

    test_model = build_model_fn(_class=Transformer,
                                **kwargs,
                                is_train=False,
                                reuse=False)

    # create session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=sess_config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # read checkpoint
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print('latest model -- {} has been loaded.'.format(ckpt))
    else:
        print('no model found!')
        exit(-1)

    writeToFile = True
    if writeToFile:
        greedy_out_filename = out_dir + 'greedy.out'
        # beam_out_filename = 'beam.out'
        greedy_out_file = open(greedy_out_filename, 'w')
        #beam_out_file = open(beam_out_filename, 'w')

    # for test_sent, answer in tqdm(test_data):
    #     greedy_pred = test_model.predict(sess, [test_sent], [np.arange(len(test_sent))], max_sent_len, beam_size=None)[0]
    #     # pred = test_model.predict(sess, [test_sent], [np.arange(len(test_sent))], max_sent_len, beam_size=4)
    #     #print('inputs:', ' '.join(trans_sent(test_sent, idx2word_vocab, mode='idx2word')))
    #     #print('answer:', ' '.join(trans_sent(answer, idx2word_vocab, mode='idx2word')))
    #     greedy_sent_list = trans_sent(greedy_pred, idx2word_vocab, mode='idx2word')
    #     #print('greedy predicts:', ' '.join(greedy_sent_list))
    #     if writeToFile:
    #         greedy_out_file.write(get_smiles(greedy_sent_list, vocab_set, sign_start, sign_end, idx2word_vocab)+'\n')
    #     #print('----------------------')
    #     #for b in range(pred.shape[1]):
    #     #    print('beam predicts:', ' '.join(trans_sent(pred[:, b, :][0], idx2word_vocab, mode='idx2word')))
    #     #    print('+++++++++++++++++++++++++++++++++++')
    #     #print('----------------------')
    #     # break

    pos = 0
    N = len(test_data)
    _s_t = time.time()
    while True:
        # get batch data
        batch_inputs, _, batch_targets, _, new_batch_size = \
            get_batch_data(test_data, pos, batch_size)

        # padd and get position
        # front padding for inputs of encoder
        batch_inputs = batch_pad(batch_inputs,
                                 val=vocab_set[sign_pad],
                                 front=False)

        batch_greedy_pred = test_model.predict(sess,
                                               batch_inputs,
                                               max_length=200)
        print('{}/{}'.format(pos + new_batch_size, N))

        for b in range(new_batch_size):
            greedy_sent_list = trans_sent(batch_greedy_pred[b, :],
                                          idx2word_vocab,
                                          mode='idx2word')
            # print(greedy_sent_list)
            greedy_out_file.write(
                get_smiles(greedy_sent_list, vocab_set, sign_start, sign_end,
                           idx2word_vocab) + '\n')

        # terminate condition
        if new_batch_size < batch_size:
            break

        # update pos
        pos += new_batch_size

    _e_t = time.time()
    print('all test done, used {:.2f} secs.'.format(_e_t - _s_t))

    if writeToFile:
        greedy_out_file.close()
        # beam_out_file.close()
