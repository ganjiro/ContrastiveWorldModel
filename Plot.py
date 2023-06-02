import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy import load

# file_path = "Writers/halfcheetah_E_test_action_recn"
# file_name = "Vanilla.npy"
#
# data = load(os.path.join(file_path,file_name), allow_pickle=True)
#
# mean = data.mean(axis=0)
# variance = data.std(axis=0)
# x = np.arange(len(mean))*5000
#
# plt.plot(x,mean)
# plt.fill_between(x, (mean-variance), (mean+variance), color='b', alpha=.1)
# plt.savefig(os.path.join(file_path,os.path.splitext(file_name)[0]+".png"))

file_path = "Writers/halfcheetah_E_test_action_recn"
file_name = "Vanilla.npy"

data = load(os.path.join(file_path, file_name), allow_pickle=True)

mean = data.mean(axis=0)
variance = data.std(axis=0)
x = np.arange(len(mean)) * 5000

plt.plot(x, mean)
plt.fill_between(x, (mean - variance), (mean + variance), color='b', alpha=.1)

file_name = "batch+action.npy"

data = load(os.path.join(file_path, file_name), allow_pickle=True)

mean = data.mean(axis=0)
variance = data.std(axis=0)
x = np.arange(len(mean)) * 5000

plt.plot(x, mean, color='g',)
plt.fill_between(x, (mean - variance), (mean + variance), color='g', alpha=.1)


file_name = "oeb.npy"

data = load(os.path.join(file_path, file_name), allow_pickle=True)

mean = data.mean(axis=0)
variance = data.std(axis=0)
x = np.arange(len(mean)) * 5000

plt.plot(x, mean, color='y',)
plt.fill_between(x, (mean - variance), (mean + variance), color='y', alpha=.1)

plt.savefig(os.path.join(file_path, "merge.png"))
