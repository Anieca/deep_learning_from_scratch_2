import numpy as np

from src.functions import softmax, cross_entropy_error, sigmoid
from src.utils import UnigramSampler


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t.argmax(axis=1) if t.shape == self.y.shape else t

        return cross_entropy_error(self.y, self.t)

    def backward(self, dout):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class SigmoidWithLoss:
    def __init_(self):
        self.params = []
        self.grads = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = sigmoid(x)
        self.t = t
        loss = cross_entropy_error(np.c_[1 - self.y, self.y], t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx


class TwoLayerNet:
    def __init__(self, in_size, hidden_size, out_size):
        i, h, o = in_size, hidden_size, out_size
        W1 = 0.01 * np.random.randn(i, h)
        b1 = np.zeros((h))
        W2 = 0.01 * np.random.randn(h, o)
        b2 = np.zeros((o))

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        y = self.predict(x)
        loss = self.loss_layer.forward(y, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class Matmul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W,  = self.params
        self.x = x
        return np.dot(x, W)

    def backward(self, dout):
        W,  = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size)
        W_out = 0.01 * np.random.randn(hidden_size, vocab_size)

        self.in_layer0 = Matmul(W_in)
        self.in_layer1 = Matmul(W_in)
        self.out_layer = Matmul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = 0.5 * (h0 + h1)

        y = self.out_layer.forward(h)
        loss = self.loss_layer.forward(y, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)

        dh = dout * target_W
        return dh


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params = []
        self.grads = []
        for embed_dot_layer in self.embed_dot_layers:
            self.params += embed_dot_layer.params
            self.grads += embed_dot_layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_targets = self.sampler.get_negative_sample(target)

        y = self.embed_dot_layers[0].forward(h, target)
        t = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(y, t)

        t = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_targets[:, i]
            y = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(y, t)

        return loss

    def backward(self, dout):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dy = l0.backward(dout)
            dh += l1.backward(dy)
        return dh


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):

        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')

        self.in_layers = []
        for i in range(window_size * 2):
            layer = Embedding(W_in)
            self.in_layers.append(layer)

        self.ns_loss = NegativeSamplingLoss(W_out, corpus)

        layers = self.in_layers + [self.ns_loss]

        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dh = self.ns_loss.backward(dout)
        dh *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dh)


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dh = dh_next * (1 - h_next ** 2)
        dh_prev = np.dot(dh, Wh.T)
        dWh = np.dot(h_prev.T, dh)
        dx = np.dot(dh, Wx.T)
        dWx = np.dot(x.T, dh)
        db = np.sum(dh, axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.stateful = stateful
        self.h = None
        self.dh = None

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dh = 0
        grads = [0, 0, 0]

        dxs = np.empty((N, T, D), dtype='f')

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, xs):
        W, = self.params
        N, T = xs.shape
        V, D = W.shape

        self.layers = []
        hs = np.empty((N, T, D), dtype='f')
        for t in range(T):
            layer = Embedding(W)
            h = layer.forward(xs[:, t])
            hs[:, t, :] = h
            self.layers.append(layer)

        return hs

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.layers = None

    def forward(self, x):
        W, b = self.params
        N, T, D = x.shape

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        W, b = self.params
        N, T, D = dout.shape
        x = self.x

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:
            ts = ts.argmax(axis=2)
        mask = (ts != self.ignore_label)

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))
        return dx


class SimpleRNNLM:
    def __init__(self, vocab_size, word_vecs, hidden_size):
        V, D, H = vocab_size, word_vecs, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype('f')
        rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')

        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
