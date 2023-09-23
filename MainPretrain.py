import os.path
import matplotlib.pyplot as plt
import numpy as np
from Manager import Manager

if __name__ == "__main__":

    env_name = "halfcheetah-medium-expert-v2"

    writer_name = "Writers/halfcheetah_test_pre_10K/"
    save_path = "Models/halfcheetah_test_pre_batch_10K"
    repeats = 3

    runs = []
    model_name = "end_to_end"
    test_name = "Test_pre"
    for i in range(repeats):
        manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False,
                          perc=0,
                          writer_name=writer_name, entire_trajectory=False, dimension=10000,
                          action=False, equal_size=True,
                          test_name=test_name)

        manager.train(300)
        # manager.load(save_path)
        runs.append(manager.test_td3_bc(corr_type=1, aug=1, hyperparameter=0.5, iterations=200000, save_path=save_path, pre_training=True))
        np.save(os.path.join(writer_name, test_name + '.npy'), np.array(runs))
    runs = np.array(runs)
    np.save(os.path.join(writer_name, test_name + '.npy'), runs)


    save_path = "Models/halfcheetah_test_pre_vanilla_10K_replay"
    runs = []
    model_name = ""
    test_name = "Vanilla"
    for i in range(repeats):
        manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True,
                          perc=0,
                          writer_name=writer_name, entire_trajectory=False, dimension=10000,
                          action=False, equal_size=True,
                          test_name=test_name)

        runs.append(manager.test_td3_bc(corr_type=1, aug=0, hyperparameter=0, iterations=200000, save_path=save_path, pre_training=True))
        np.save(os.path.join(writer_name, test_name + '.npy'), np.array(runs))
    runs = np.array(runs)
    np.save(os.path.join(writer_name, test_name + '.npy'), runs)

