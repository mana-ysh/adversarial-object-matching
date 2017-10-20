
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, dim):
        self.dim = dim
        super(Generator, self).__init__(
            W = L.Linear(dim, dim)
        )

    def __call__(self, src_emb):
        """
        simple projection
        """
        src_emb = chainer.Variable(src_emb)
        return self.W(src_emb)


class Discriminator(chainer.Chain):
    def __init__(self, dim, hidden=2048, drop_rate=0.1):
        self.dim = dim
        self.hidden = hidden
        self.drop_rate = drop_rate
        super(Discriminator, self).__init__(
            W1 = L.Linear(dim, hidden),
            W2 = L.Linear(hidden, 2)
        )

    def __call__(self, xs):
        # TODO: adding noise?
        xs = F.dropout(xs, ratio=self.drop_rate)
        hs = F.leaky_relu(self.W1(xs))
        return self.W2(hs)
