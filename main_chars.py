import numpy as np
import pandas as pd


class DataReader:
    def __init__(self, path, seq_length):
        #readl all data form csv
        self.fp = pd.read_csv(path, header=None)
        # Combine all rows into a single string
        self.data = ''.join(self.fp[0].astype(str).values)
        # find unique chars
        chars = list(set(self.data))
        # create dictionary mapping for each char
        self.char_to_ix = {ch: i for (i, ch) in enumerate(chars)}
        self.ix_to_char = {i: ch for (i, ch) in enumerate(chars)}
        # total data
        self.data_size = len(self.data)
        # num of unique chars
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start: input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start + 1: input_end + 1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            # reset pointer
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0



class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        #Weight ,bias
        self.U = np.random.randn(hidden_size, vocab_size) * 0.01
        self.V = np.random.randn(vocab_size, hidden_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))

        # memory vars for adagrad,
        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)

    def softmax(self, x):
        p = np.exp(x - np.max(x))
        return p / np.sum(p)

    def forward(self, inputs, hprev):
        xs, hs, os, ycap = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.U, xs[t]) + np.dot(self.W, hs[t - 1]) + self.b)
            os[t] = np.dot(self.V, hs[t]) + self.c
            ycap[t] = self.softmax(os[t])
        return xs, hs, ycap

    def backward(self, xs, hs, ps, targets):
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dV += np.dot(dy, hs[t].T)
            dc += dy
            dh = np.dot(self.V.T, dy) + dhnext
            dhrec = (1 - hs[t] * hs[t]) * dh
            db += dhrec
            dU += np.dot(dhrec, xs[t].T)
            dW += np.dot(dhrec, hs[t - 1].T)
            dhnext = np.dot(self.W.T, dhrec)
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -1, 1, out=dparam)
        return dU, dW, dV, db, dc

    def loss(self, ps, targets):
        return sum(-np.log(ps[t][targets[t], 0]) for t in range(self.seq_length))
#update model using Adagrad optimization algorithm.
    def update_model(self, dU, dW, dV, db, dc):
        for param, dparam, mem in zip([self.U, self.W, self.V, self.b, self.c],
                                      [dU, dW, dV, db, dc],
                                      [self.mU, self.mW, self.mV, self.mb, self.mc]):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            y = np.dot(self.V, h) + self.c
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

 #   Generates a sequence of characters from the model.
    def train(self, data_reader):
        iter_num = 0
        threshold = 0.1
        smooth_loss = -np.log(1.0 / data_reader.vocab_size) * self.seq_length
        while smooth_loss > threshold:

            if data_reader.just_started():
                hprev = np.zeros((self.hidden_size, 1))
            inputs, targets = data_reader.next_batch()
            xs, hs, ps = self.forward(inputs, hprev)
            dU, dW, dV, db, dc = self.backward(xs, hs, ps, targets)
            loss = self.loss(ps, targets)
            self.update_model(dU, dW, dV, db, dc)
            smooth_loss = smooth_loss * 0.99 + loss * 0.01
            hprev = hs[self.seq_length - 1]
            if iter_num % 500 == 0:

                sample_ix = self.sample(hprev, inputs[0], 200)
                print(''.join(data_reader.ix_to_char[ix] for ix in sample_ix))
                print("\n\niter :%d, loss:%f" % (iter_num, smooth_loss))
                print(smooth_loss)
                print(threshold)
            iter_num += 1


def run():
    seq_length = 10
    data_reader = DataReader("myDataset.csv", seq_length)
    rnn = RNN(hidden_size=100, vocab_size=data_reader.vocab_size, seq_length=seq_length, learning_rate=0.1)
    rnn.train(data_reader)
    print(rnn.predict(data_reader, 'get', 10))


if __name__ == "__main__":
    run()

