from __future__ import print_function
import tensorflow as tf
import json
import riseml

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def sample(data):
    text = ''
    args = json.loads(data.decode("utf-8"))
    with open(os.path.join(args['save_dir'], 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args['save_dir'], 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            text = model.sample(sess, chars, vocab, int(args['n']), args['prime'],
                               int(args['sample']))
    return json.dumps({
        'text': text,
    }).encode("utf-8")

if __name__ == '__main__':
    sample('{"n": 500, "save_dir": "./save", "sample": 1, "prime": " "}'.encode('utf-8'))
    sample('{"n": 500, "save_dir": "./save", "sample": 1, "prime": " "}'.encode('utf-8'))
    #riseml.serve(sample)

