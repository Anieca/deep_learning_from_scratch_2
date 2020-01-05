import pickle
import numpy as np

from src.layers import Affine, Sigmoid, Matmul, Embedding
from src.layers import SoftmaxWithLoss, NegativeSamplingLoss
from src.time_layers import TimeAffine, TimeEmbedding, TimeRNN, TimeLSTM
from src.time_layers import TimeDropout, TimeSoftmaxWithLoss


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


class RNNLM:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype('f')
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params = []
        self.grads = []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        ys = self.predict(xs)
        loss = self.loss_layer.forward(ys, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, filename='RNN_params.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, filename='RNN_params.pkl'):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)


class BetterRNNLM:
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650,
                 dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype('f')
        lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')

        lstm_Wx2 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        ys = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(ys, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

    def save_params(self, filename='RNN_params.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, filename='RNN_params.pkl'):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
