from datasets import sequence
from src.models import AttentionSeq2Seq
from src.optimizers import Adam
from src.trainer import Trainer
from src.utils import eval_seq2seq


def train_attention_seq2seq():
    train, test = sequence.load_data('date.txt')
    x_train, t_train = train
    x_test, t_test = test
    char_to_id, id_to_char = sequence.get_vocab()

    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 256
    batch_size = 128
    max_epoch = 10
    max_grad = 5.0

    model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
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
                model, question, correct, id_to_char, verbose, True)

        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print(f'val acc {100*acc:.3f}%')


if __name__ == '__main__':
    train_attention_seq2seq()
