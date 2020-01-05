from datasets import ptb
from src.optimizers import SGD
from src.models import RNNLM, BetterRNNLM
from src.trainer import RNNLMTrainer
from src.utils import eval_perplexity


def train_rnnlm():
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100
    time_size = 35

    lr = 20.0
    max_epoch = 4
    max_grad = 0.25

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)

    xs = corpus[:-1]
    ts = corpus[1:]

    model = RNNLM(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    trainer = RNNLMTrainer(model, optimizer)
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad)
    trainer.plot(ylim=(0, 500))

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print(f'test perplexity: {ppl_test}')
    model.save_params()


def train_better_rnnlm():
    batch_size = 20
    wordvec_size = 650
    hidden_size = 650
    time_size = 35

    lr = 20.0
    max_epoch = 40
    max_grad = 0.25
    dropout = 0.5

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)

    xs = corpus[:-1]
    ts = corpus[1:]

    model = BetterRNNLM(vocab_size, wordvec_size, hidden_size, dropout)
    optimizer = SGD(lr)

    trainer = RNNLMTrainer(model, optimizer)

    best_ppl = float('inf')
    for epoch in range(max_epoch):
        trainer.fit(xs, ts, 1, batch_size, time_size, max_grad)

        model.reset_state()
        ppl = eval_perplexity(model, corpus_test)
        print(f'test perplexity: {ppl}')

        if ppl > best_ppl:
            best_ppl = ppl
            model.save_params('better_RNN_params.pkl')
        else:
            lr /= 4.0
            optimizer.lr = lr
        model.reset_state()


if __name__ == '__main__':
    train_rnnlm()
    train_better_rnnlm()
