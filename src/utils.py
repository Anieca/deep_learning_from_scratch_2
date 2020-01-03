import collections
import numpy as np


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and \
                        np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads


def preprocess(text):
    text = text.lower().replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            word_id = len(word_to_id)
            word_to_id[word] = word_id
            id_to_word[word_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_sililar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query} is not found.')
        return

    print(f'\n[query] {query}')

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    for i in (-1 * similarity).argsort()[:top+1]:
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]} : {similarity[i]}')


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype='f')
    N = np.sum(C)
    S = np.sum(C, axis=1)

    cnt = 0
    total = C.shape[0] * C.shape[1]

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(N * C[i, j] / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
            cnt += 1

            if verbose and cnt % (total // 100) == 0:
                print(f'{100 * cnt / total:.1f}% done')

    return M


def create_contexts_target(corpus, window_size=1):

    context_list = []
    target_list = corpus[window_size: -window_size]
    for i in range(window_size, len(corpus) - window_size):
        contexts = []
        for j in range(-window_size, window_size + 1):
            if j == 0:
                continue
            contexts.append(corpus[i+j])
        context_list.append(contexts)

    return np.array(context_list), np.array(target_list)


def convert_one_hot(corpus, vocab_size):

    if corpus.ndim == 1:
        one_hot = np.zeros((corpus.shape[0], vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        N, C = corpus.shape
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx1, word_ids in enumerate(corpus):
            for idx2, word_id in enumerate(word_ids):
                one_hot[idx1, idx2, word_id] = 1

    return one_hot


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for ward_id in corpus:
            counts[ward_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros(
            (batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()

            negative_sample[i, :] = np.random.choice(
                self.vocab_size, self.sample_size, replace=False, p=p)

        return negative_sample


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print(f'{word} is not found.')
            return

    print(f'\n[analogy] {a}:{b} = {c}:?')

    a_vec = word_matrix[word_to_id[a]]
    b_vec = word_matrix[word_to_id[b]]
    c_vec = word_matrix[word_to_id[c]]

    query_vec = normalize(b_vec - a_vec + c_vec)
    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print(f'{answer}:{np.dot(word_matrix[word_to_id[answer]], query_vec)}')

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(f'{id_to_word[i]}:{similarity[i]}')

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
