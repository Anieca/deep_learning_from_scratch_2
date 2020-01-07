import numpy as np

from datasets import ptb, sequence
from src.models import RNNLM, BetterRNNLM, Seq2Seq, PeekySeq2Seq
from src.trainer import Trainer
from src.optimizers import Adam
from src.functions import softmax
from src.utils import eval_seq2seq


class GenRNNLM(RNNLM):
    def generate(self, start_id, skip_ids, sample_size=100):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            y = self.predict(x)
            p = softmax(y.flatten())

            x = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (x not in skip_ids):
                word_ids.append(int(x))

        return word_ids


class GenBetterRNNLM(BetterRNNLM):
    def generate(self, start_id, skip_ids, sample_size=100):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            y = self.predict(x)
            p = softmax(y.flatten())

            x = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (x not in skip_ids):
                word_ids.append(int(x))

        return word_ids


def generate_text():
    corpus, word_to_id, id_to_word = ptb.load_data('train')

    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[word] for word in skip_words]

    print('-'*50)
    model = GenRNNLM()
    model.load_params()
    word_ids = model.generate(start_id, skip_ids)
    text = ' '.join([id_to_word[word_id] for word_id in word_ids])
    text = text.replace(' <eos> ', '.\n')
    print(text)

    print('-'*50)
    model = GenBetterRNNLM()
    model.load_params()
    word_ids = model.generate(start_id, skip_ids)
    text = ' '.join([id_to_word[word_id] for word_id in word_ids])
    text = text.replace(' <eos> ', '.\n')
    print(text)

    print('-'*50)
    model.reset_state()
    for word in ('the meaning of life is').split():
        x = np.array(word_to_id[word]).reshape(1, 1)
        if word == 'is':
            start_id = word_to_id[word]
            word_ids = model.generate(start_id, skip_ids)
        else:
            model.predict(x)
    print('the meaning of life is ?')
    text = ' '.join([id_to_word[word_id] for word_id in word_ids[1:]])
    text = text.split('<eos>')[0]
    print(text)


def toy_problem(reverse=False, peeky=False):
    (x_train, t_train), (x_test, t_test) = sequence.load_data()
    if reverse:
        x_train = x_train[:, ::-1]
        x_test = x_test[:, ::-1]
    char_to_id, id_to_char = sequence.get_vocab()

    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 128
    batch_size = 128
    max_epoch = 25
    max_grad = 5.0

    if peeky:
        model = PeekySeq2Seq(vocab_size, wordvec_size, hidden_size)
    else:
        model = Seq2Seq(vocab_size, wordvec_size, hidden_size)

    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, 1, batch_size, max_grad)

        correct_num = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(
                model, question, correct, id_to_char, verbose, reverse)

        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print(f'val acc {100*acc:.3f}%')


if __name__ == '__main__':
    # generate_text()
    # toy_problem()
    # toy_problem(reverse=True)
    toy_problem(reverse=True, peeky=True)
