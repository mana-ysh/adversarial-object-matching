
import chainer
import chainer.functions as F
import numpy as np


class GANTrainer(object):
    def __init__(self, models, opts, logger, valid_data, epoch, batchsize):
        self.gen, self.dis = models
        self.gen_opt, self.dis_opt = opts
        self.logger = logger
        self.epoch = epoch
        self.batchsize = batchsize
        self.valid_data = valid_data

    def fit(self, src_emb, trg_emb):
        n_src_word = len(src_emb)
        n_trg_word = len(trg_emb)
        n_sample = max(n_src_word, n_trg_word)
        for epoch in range(self.epoch):
            self.logger.info('start {} epoch'.format(epoch+1))
            sum_gen_loss = 0
            sum_dis_loss = 0
            for i in range(n_sample // self.batchsize):
                src_idxs = np.random.randint(0, n_src_word, size=self.batchsize)
                trg_idxs = np.random.randint(0, n_trg_word, size=self.batchsize)
                _src_emb = src_emb[src_idxs]
                _trg_emb = trg_emb[trg_idxs]

                prj_src_emb = self.gen(_src_emb)
                src_ys = self.dis(prj_src_emb)
                gen_loss = F.softmax_cross_entropy(src_ys, chainer.Variable(np.zeros(self.batchsize, dtype=np.int32)))
                dis_loss = F.softmax_cross_entropy(src_ys, chainer.Variable(np.ones(self.batchsize, dtype=np.int32)))

                trg_ys = self.dis(_trg_emb)
                dis_loss += F.softmax_cross_entropy(trg_ys, chainer.Variable(np.zeros(self.batchsize, dtype=np.int32)))

                # update for generator
                self.gen.zerograds()
                gen_loss.backward()
                self.gen_opt.update()

                # update for discriminator
                self.dis.zerograds()
                dis_loss.backward()
                self.dis_opt.update()

                sum_gen_loss += gen_loss.data
                sum_dis_loss += dis_loss.data

            self.logger.info('    generator loss: {}'.format(sum_gen_loss))
            self.logger.info('discriminator loss: {}'.format(sum_dis_loss))
