import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy import load


def plot(file_path, file_name_vanilla, file_name_mine, file_name_nopre, save_name):
    data = load(os.path.join(file_path, file_name_vanilla + '.npy'), allow_pickle=True)[:,:-20]
    # # data = np.delete(data, 1, axis=0)
    # # data = np.delete(data, 1, axis=0)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)

    mean = data.mean(axis=0)
    variance = data.std(axis=0)
    x = np.arange(len(mean)) * 5000

    plt.plot(x, mean, color='b', label="TD3BC")
    plt.fill_between(x, (mean - variance), (mean + variance), color='b', alpha=.1)

    data = load(os.path.join(file_path, file_name_mine + '.npy'), allow_pickle=True)[:,:-20]
    # data = np.delete(data, 1, axis=0)
    # data = np.delete(data, 0, axis=0)

    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)
    data = np.delete(data, 40, 1)

    mean = data.mean(axis=0)
    variance = data.std(axis=0)
    x = np.arange(len(mean)) * 5000

    plt.plot(x, mean, color='g', label="Batch_Aug")
    plt.fill_between(x, (mean - variance), (mean + variance), color='g', alpha=.1)

    data = load(os.path.join(file_path, file_name_nopre + '.npy'), allow_pickle=True)[:,:-21]
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 1)


    mean = data.mean(axis=0)
    variance = data.std(axis=0)
    x = (np.arange(len(mean)) * 5000) + 200000

    plt.plot(x, mean, color='r', label="TD3")
    plt.fill_between(x, (mean - variance), (mean + variance), color='r', alpha=.1)

    plt.legend(loc="upper left")
    plt.axvline(x=200000, color='black')

    plt.savefig(os.path.join(file_path, save_name + ".pdf"), format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    name = "halfcheetah_test_pre_10K_replay"  #Writers/halfcheetah_test_pre/Test_pre.npy
    writer_name = "Writers/" + name

    test_name_1 = "Test_pre"
    test_name_2 = "Vanilla"
    test_name_3 = "half_td3"
    plot(writer_name, test_name_2, test_name_1,test_name_3, name)
