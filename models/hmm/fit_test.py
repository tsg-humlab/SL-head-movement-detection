import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
from random import randint
import pickle


def main():
    x = [torch.randn(1, randint(10, 100), 5) for _ in range(200)]

    with open('hmm.p', 'rb') as input_handle:
        model = pickle.load(input_handle)
    # model = DenseHMM([Normal(), Normal(), Normal()], max_iter=5, verbose=True)
    # model.fit(x)
    print(model.distributions[0].means)

    with open('hmm.p', 'wb') as output_handle:
        pickle.dump(model, output_handle)

    pass


if __name__ == '__main__':
    main()
