import numpy as np

from Manager import Manager

if __name__ == "__main__":

    env_name = "halfcheetah-medium-replay-v2"

    writer_name = "Writers/halfcheetah_alldataset"
    save_path = "Models/halfcheetah_alldataset"

    # model_name = ""
    # test_name = "Vanilla"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
    #                   test_name=test_name)
    # manager.test_td3_bc(1)

    # model_name = "end_to_end"
    # for i in np.arange(0.5, 1.0, 0.1):
    #     test_name = "S4RL_" + str(i)
    #     manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                       writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
    #                       test_name=test_name)
    #     # manager.train(100)
    #     manager.load(save_path)
    #     manager.test_td3_bc(corr_type=1, aug=4, hyperparameter=i)

    model_name = "end_to_end"
    test_name = "batch"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None, all_dataset=False,
                      test_name=test_name)
    # manager.train(200)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=1)

    model_name = "end_to_end"
    test_name = "OEB"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None, all_dataset=False,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=2)


    model_name = "end_to_end"
    test_name = "S4rl"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None, all_dataset=True,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=3)

    model_name = "end_to_end"
    test_name = "Action"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None, all_dataset=True,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=9)
