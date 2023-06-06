import os.path
import matplotlib.pyplot as plt
import numpy as np
from Manager import Manager

#
# def plot(data, file_path, file_name):
#     mean = data.mean(axis=0)
#     variance = data.std(axis=0)
#     x = np.arange(len(mean)) * 5000
#
#     plt.plot(x, mean)
#     plt.fill_between(x, (mean - variance), (mean + variance), color='b', alpha=.1)
#     plt.savefig(os.path.join(file_path, os.path.splitext(file_name)[0] + ".png"))


if __name__ == "__main__":

    env_name = "walker2d-medium-expert-v2"

    writer_name = "Writers/walker2d_Fill/"
    save_path = "Models/walker2d_Fill"
    repeats = 5    #
    runs = []
    model_name = "end_to_end"
    test_name = "batch"
    for i in range(repeats):
        manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                          writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=50000,
                          action=False, equal_size=True,
                          test_name=test_name)

        manager.train(200)
        runs.append(manager.test_td3_bc(corr_type=1, aug=1, hyperparameter=0.5, iterations=500000))
        np.save(os.path.join(writer_name, test_name + '.npy'), np.array(runs))
    runs = np.array(runs)
    np.save(os.path.join(writer_name, test_name + '.npy'), runs)
    # plot(runs, save_path, test_name + '.npy')

    # runs = []
    # model_name = "end_to_end"
    # test_name = "oeb"
    # for i in range(repeats):
    #     manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
    #                       writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=500000,
    #                       action=True,
    #                       test_name=test_name)
    #
    #     manager.train(300)
    #     runs.append(manager.test_td3_bc(corr_type=1, aug=2, hyperparameter=1, iterations=500000))
    # runs = np.array(runs)
    # np.save(os.path.join(writer_name, test_name + '.npy'), runs)

    runs = []
    model_name = ""
    test_name = "Vanilla"
    for i in range(repeats):
        manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                          writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=500000,
                          test_name=test_name)
        runs.append(manager.test_td3_bc(corr_type=1, iterations=1000000))
        np.save(os.path.join(writer_name, test_name + '.npy'), np.array(runs))
    runs = np.array(runs)
    np.save(os.path.join(writer_name, test_name + '.npy'), runs)
    # plot(runs, save_path, test_name + '.npy')

    # model_name = "end_to_end"
    # for i in np.arange(0.2, 1.0, 0.1):
    #     test_name = "batch"+str(i)
    #     manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
    #                       writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=500000,
    #                       test_name=test_name)
    #     # manager.train(100)
    #     manager.load(save_path)
    #     manager.test_td3_bc(corr_type=1, aug=1, hyperparameter=i)

    # model_name = "end_to_end"
    # test_name = "OEB"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
    #                   test_name=test_name)
    # manager.load(save_path)
    # manager.test_td3_bc(corr_type=1, aug=2)
    #
    #
    # model_name = "end_to_end"
    # test_name = "S4rl"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
    #                   test_name=test_name)
    # manager.load(save_path)
    # manager.test_td3_bc(corr_type=1, aug=3)
    #
    # model_name = "end_to_end"
    # test_name = "Action"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
    #                   test_name=test_name)
    # manager.load(save_path)
    # manager.test_td3_bc(corr_type=1, aug=9)
