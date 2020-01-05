from datasets.spiral import load_data
from src.optimizers import SGD
from src.models import TwoLayerNet
from src.trainer import Trainer


def main():
    max_epoch = 300
    batch_size = 30

    x, t = load_data()
    model = TwoLayerNet(in_size=2, hidden_size=10, out_size=3)
    optimizer = SGD(1.0)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, max_epoch, batch_size, None, 10)
    trainer.plot()


if __name__ == '__main__':
    main()
