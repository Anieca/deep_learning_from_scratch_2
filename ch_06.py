from datasets import ptb
from src.optimizers import SGD
from src.models import RNNLM
from src.trainer import RNNLMTrainer
from src.utils import eval_perplexity


def main():
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


if __name__ == '__main__':
    main()
