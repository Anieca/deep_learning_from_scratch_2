from sklearn.utils.extmath import randomized_svd

from datasets import ptb
from src.utils import create_co_matrix, ppmi, most_sililar


def main():

    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    C = create_co_matrix(corpus, vocab_size, window_size=window_size)
    W = ppmi(C, verbose=True)
    U, S, V = randomized_svd(W, wordvec_size, n_iter=5, random_state=None)

    queries = ['you', 'year', 'car', 'toyota']
    for query in queries:
        most_sililar(query, word_to_id, id_to_word, U, top=5)


if __name__ == '__main__':
    main()
