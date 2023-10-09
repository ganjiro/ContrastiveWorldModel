import os.path
import matplotlib.pyplot as plt
import numpy as np
from Manager import Manager

if __name__ == "__main__":

    env_name = ["walker2d-medium-replay-v2", "walker2d-random-v2"]  # "halfcheetah-medium-expert-v2",

    writer_name = ["test_500K/walker2d_MR/", "test_500K/walker2d_R/"]  # "test_500K/halfcheetah_ME/",
    eql_size = [False, False] #True
    save_path = "Models/walker_test_pre_batch_10K_random"
    repeats = 5
    for i in range(len(env_name)):
        runs = []
        model_name = "end_to_end"
        test_name = "Test_pre"
        for _ in range(repeats):
            manager = Manager(model_name=model_name, env_name=env_name[i], savepath=save_path, contrastive=False,
                              perc=0,
                              writer_name=writer_name[i], entire_trajectory=False, dimension=10000,
                              action=False, equal_size=eql_size[i],
                              test_name=test_name)

            manager.train(300)

            runs.append(
                manager.test_td3_bc(corr_type=1, aug=1, hyperparameter=0.5, iterations=200000, save_path=save_path,pre_training=True))
            np.save(os.path.join(writer_name[i], test_name + '.npy'), np.array(runs))
        runs = np.array(runs)
        np.save(os.path.join(writer_name[i], test_name + '.npy'), runs)

        save_path = "Models/walker_test_pre_vanilla_10K_random"
        runs = []
        model_name = ""
        test_name = "Vanilla"
        for _ in range(repeats):
            manager = Manager(model_name=model_name, env_name=env_name[i], savepath=save_path, contrastive=True,
                              perc=0,
                              writer_name=writer_name[i], entire_trajectory=False, dimension=10000,
                              action=False, equal_size=eql_size[i],
                              test_name=test_name)

            runs.append(
                manager.test_td3_bc(corr_type=1, aug=0, hyperparameter=0, iterations=200000, save_path=save_path,
                                    pre_training=True))
            np.save(os.path.join(writer_name[i], test_name + '.npy'), np.array(runs))
        runs = np.array(runs)
        np.save(os.path.join(writer_name[i], test_name + '.npy'), runs)
