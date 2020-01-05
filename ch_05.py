from datasets import ptb
from src.optimizers import SGD
from src.models import SimpleRNNLM
from src.trainer import RNNLMTrainer


def main():
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5

    lr = 0.1
    max_epoch = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]
    ts = corpus[1:]

    model = SimpleRNNLM(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    trainer = RNNLMTrainer(model, optimizer)
    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()


if __name__ == '__main__':
    main()
