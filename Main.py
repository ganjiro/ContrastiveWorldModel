import os.path
import matplotlib.pyplot as plt
import numpy as np
from Manager import Manager

if __name__ == "__main__":

    env_name = "walker2d-medium-expert-v2"
    for d in [3, 5]:
        writer_name = "Writers/walker2d_ME_fill_50K_" + str(d) + "0/"
        save_path = "Models/walker2d_ME_fill_50K_" + str(d) + "0"
        repeats = 5

        runs = []
        model_name = "end_to_end"
        test_name = "batch"
        for i in range(repeats):
            manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False,
                              perc=d * 1e-1,
                              writer_name=writer_name, entire_trajectory=False, dimension=50000,
                              action=False, equal_size=True,
                              test_name=test_name)

            manager.train(500)
            runs.append(manager.test_td3_bc(corr_type=0, aug=0, hyperparameter=0, iterations=500000))
            np.save(os.path.join(writer_name, test_name + '.npy'), np.array(runs))
        runs = np.array(runs)
        np.save(os.path.join(writer_name, test_name + '.npy'), runs)

        runs = []
        model_name = ""
        test_name = "Vanilla"
        for i in range(repeats):
            manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True,
                              perc=d * 1e-1,
                              writer_name=writer_name, entire_trajectory=False, dimension=50000,
                              equal_size=True,
                              test_name=test_name)
            runs.append(manager.test_td3_bc(corr_type=4, iterations=500000))
        runs = np.array(runs)
        np.save(os.path.join(writer_name, test_name + '.npy'), runs)

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
