import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy import load


def plot(file_path, file_name_vanilla, file_name_mine, save_name):
    data = load(os.path.join(file_path, file_name_vanilla + '.npy'), allow_pickle=True)

    mean = data.mean(axis=0)
    variance = data.std(axis=0)
    x = np.arange(len(mean)) * 5000

    plt.plot(x, mean, color='b', label=file_name_vanilla)
    plt.fill_between(x, (mean - variance), (mean + variance), color='b', alpha=.1)

    data = load(os.path.join(file_path, file_name_mine + '.npy'), allow_pickle=True)

    mean = data.mean(axis=0)
    variance = data.std(axis=0)
    x = np.arange(len(mean)) * 5000

    plt.plot(x, mean, color='g', label=file_name_mine)
    plt.fill_between(x, (mean - variance), (mean + variance), color='g', alpha=.1)
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(file_path, save_name + ".png"))
