import matplotlib.pyplot as plt
import numpy as np
import seaborn
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM


def main():
    seaborn.set_style('whitegrid')
    np.random.seed(0)

    d1 = Categorical([[0.25, 0.25, 0.25, 0.25]])
    d2 = Categorical([[0.10, 0.40, 0.40, 0.10]])

    model = DenseHMM([d1, d2],
                     edges=[[0.89, 0.1], [0.1, 0.9]],
                     starts=[0.5, 0.5],
                     ends=[0.01, 0.0])

    sequence = 'CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC'
    x = np.array([[[['A', 'C', 'G', 'T'].index(char)] for char in sequence]])

    y_hat = model.predict(x)

    print("sequence: {}".format(''.join(sequence)))
    print("hmm pred: {}".format(''.join([str(y.item()) for y in y_hat[0]])))

    plt.plot(model.predict_proba(x)[0], label=['background', 'CG island'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
