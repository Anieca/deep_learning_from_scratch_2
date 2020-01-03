import matplotlib.pyplot as plt
import numpy as np
from time import time

from src.utils import remove_duplicate, clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch, batch_size, max_grad, eval_interval=20):
        self.eval_interval = eval_interval

        data_size = len(x)
        max_iters = data_size // batch_size
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time()

        for epoch in range(max_epoch):
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size: (iters+1)*batch_size]
                batch_t = t[iters*batch_size: (iters+1)*batch_size]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and iters % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time() - start_time
                    print('| epoch %d | iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1,
                             max_iters, elapsed_time,  avg_loss))
                    self.loss_list.append(avg_loss)
                    total_loss = 0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()
