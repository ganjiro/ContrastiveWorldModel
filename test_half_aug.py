import os.path
import numpy as np
from Manager import Manager
from Plot import plot

if __name__ == "__main__":

    env_name = "halfcheetah-medium-expert-v2"

    names = ["halfcheetah_ME_fill_50_30", "halfcheetah_ME_fill_50_50"]
    dimension = [50000, 50000]
    corruption = [0.3, 0.5]
    iteration = [500000, 500000]
    epochs = [150, 150]
    repeats = 5

    for j, name in enumerate(names):

        writer_name = "Final/recon/traj" + name
        save_path = "Models/" + name

        # runs = []
        # model_name = "end_to_end"
        # test_name_1 = "fill"
        # for _ in range(repeats):
        #     manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=corruption[j],
        #                       writer_name=writer_name, entire_trajectory=True, dimension=dimension[j],
        #                       action=False, equal_size=True, inject_noise=False,
        #                       test_name=test_name_1)
        #
        #     manager.train(epochs[j])
        #     runs.append(manager.test_td3_bc(corr_type=0, aug=0, hyperparameter=0, iterations=iteration[j]))
        #     np.save(os.path.join(writer_name, test_name_1 + '.npy'), np.array(runs))
        # runs = np.array(runs)
        # np.save(os.path.join(writer_name, test_name_1 + '.npy'), runs)

        runs = []
        model_name = ""
        test_name_2 = "Remove"
        for _ in range(repeats):
            manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=corruption[j],
                              writer_name=writer_name, entire_trajectory=True, dimension=dimension[j],
                              equal_size=True, inject_noise=False,
                              test_name=test_name_2)
            runs.append(manager.test_td3_bc(corr_type=1, iterations=iteration[j]))
            np.save(os.path.join(writer_name, test_name_2 + '.npy'), np.array(runs))
        runs = np.array(runs)
        np.save(os.path.join(writer_name, test_name_2 + '.npy'), runs)

        # plot(writer_name, test_name_1, test_name_2, name)
