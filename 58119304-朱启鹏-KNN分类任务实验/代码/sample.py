import numpy as np

np.random.seed(0)
def sample(train):
    train_0 = []
    train_1 = []
    for i in range(train.shape[0]):
        if train[i,-1] == 0:
            train_0.append(train[i])
        else:
            train_1.append(train[i])
    train_0=np.array(train_0)
    train_1=np.array(train_1)
    rank_0 = np.random.permutation(range(train_0.shape[0]))
    rank_1 = np.random.permutation(range(train_1.shape[0]))
    sample_0 = []
    sample_1 = []
    sample_train = []
    for i in range(train_0.shape[0]//127):
        sample_0.append(train_0[rank_0[i * 127: (i + 1) * 127]])

    for i in range(train_1.shape[0]//48):
        sample_1.append(train_1[rank_1[i * 48:(i + 1) * 48]])
    sample_0 = np.array(sample_0)
    sample_1 = np.array(sample_1)
    rank_0 = np.random.permutation(range(3))
    rank_1 = np.random.permutation(range(3))

    for i in range(3):
        rankk = np.random.permutation(range(175))
        t = np.vstack([sample_0[rank_0[i]],sample_1[rank_1[i]]])
        t = t[rankk]
        sample_train.append(t)
    sample_train = np.array(sample_train)

    return sample_train
