import numpy as np
import pickle

from datasets import ptb
from src.links import CBOW
from src.optimizers import Adam
from src.trainer import Trainer
from src.utils import create_contexts_target, most_sililar, analogy


def train():
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size, None)
    trainer.plot()

    word_vecs = model.word_vecs

    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'

    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)


def evaluate():
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)

    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

    queries = ['you', 'year', 'car', 'toyota']
    for query in queries:
        most_sililar(query, word_to_id, id_to_word, word_vecs)

    opt = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'word_matrix': word_vecs
    }

    print('-'*50)
    analogy('king', 'man', 'queen', **opt)
    analogy('take', 'took', 'go', **opt)
    analogy('car', 'cars', 'child', **opt)
    analogy('good', 'better', 'bad', **opt)


if __name__ == '__main__':
    # train()
    evaluate()
