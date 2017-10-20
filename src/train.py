
import argparse
from chainer import optimizers
from datetime import datetime
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split

from net import Generator, Discriminator
from trainer import GANTrainer
from utils import gen_synthetic_data


np.random.seed(46)
DIM = 3000
DIM_EMB = 300
NUM = 10000
DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))


def train(args):
    # setting for logging
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.log, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{:>10} -----> {}'.format(arg, val))

    x, y = gen_synthetic_data(DIM, DIM_EMB, NUM)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

    gen = Generator(DIM_EMB)
    dis = Discriminator(DIM_EMB)

    gen_opt = optimizers.Adam()
    dis_opt = optimizers.Adam()

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    trainer = GANTrainer((gen, dis), (gen_opt, dis_opt), logger, (valid_x, valid_y), args.epoch)
    trainer.fit(train_x, train_y)


if __name__ == '__main__':
    p = argparse.ArgumentParser("Adversarial object matching with synthetic data")
    p.add_argument('--src', type=str, help='embedding file of source language')
    p.add_argument('--trg', type=str, help='embedding file of target language')
    p.add_argument('--epoch', type=int, default=100, help='number of epoch')
    p.add_argument('--batchsize', type=int, default=32, help='minibatch size')
    p.add_argument('--log', type=str, default=DEFAULT_LOG_DIR, help='log dir')

    train(p.parse_args())
