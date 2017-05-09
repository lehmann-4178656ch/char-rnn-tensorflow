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

    args = parser.parse_args()
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)

    def run(data):
        data = json.loads(data.decode("utf-8"))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                text = model.sample(sess, chars, vocab, int(data['n']), data['prime'],
                                   int(data['sample']))
        return json.dumps({
            'text': text,
        }).encode("utf-8")
    return run

if __name__ == '__main__':
    run = main()
    data = '{"n": 500, "sample": 0, "prime": " "}'.encode('utf-8')
    print(run(data).decode('utf-8'))
    print()
    data = '{"n": 500, "sample": 1, "prime": " "}'.encode('utf-8')
    print(run(data).decode('utf-8'))
    print()
    data = '{"n": 500, "sample": 2, "prime": " "}'.encode('utf-8')
    print(run(data).decode('utf-8'))
    #riseml.serve(sample)

