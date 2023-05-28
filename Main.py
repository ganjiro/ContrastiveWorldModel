import numpy as np

from Manager import Manager

if __name__ == "__main__":

    env_name = "halfcheetah-medium-expert-v2"

    writer_name = "Writers/halfcheetah_ME_test_miss/"
    save_path = "Models/halfcheetah_ME_test_miss"

    model_name = ""
    test_name = "Vanilla7"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=500000,
                      test_name=test_name)
    manager.test_td3_bc(1)

    model_name = "end_to_end"
    test_name = "batch7_decay"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=500000,
                      test_name=test_name)
    manager.train(150)
    #manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=1, hyperparameter=0.5)

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



