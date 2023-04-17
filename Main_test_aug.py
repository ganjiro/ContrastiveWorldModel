from torch import save

from Manager import Manager

if __name__ == "__main__":
    env_name = "bullet-halfcheetah-medium-expert-v0"

    writer_name = "Writers/test_idea_mse_bullet_100k/"
    save_path = "Models/OEBAug_ME_bullet_100kRand"

    model_name = "end_to_end"
    test_name = "OEB"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=False, target=True, entire_trajectory=False, dimension=100000,
                      test_name=test_name)

    #manager.load(save_path)
    manager.train(1000)
    manager.test_td3_bc(corr_type=1, aug=2)

    model_name = "end_to_end"
    test_name = "batch_aug"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=False, target=True, entire_trajectory=False, dimension=100000,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=1)

    model_name = ""
    test_name = "Noise"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=False, target=False, entire_trajectory=False, dimension=100000,
                      test_name=test_name)
    manager.test_td3_bc(corr_type=1, aug=3)

    model_name = ""
    test_name = "Vanilla"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=False, target=False, entire_trajectory=False, dimension=100000,
                      test_name=test_name)
    manager.test_td3_bc(1)

    ###############################

    # env_name = "halfcheetah-medium-expert-v2"
    #
    # writer_name = "Writers/test_aug_13_4_mujoco/"
    # save_path = "Models/OEBAug_ME_mujoco_50kRand"
    #
    # model_name = "end_to_end"
    # test_name = "OEB"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, target=True, entire_trajectory=False, dimension=50000,
    #                   test_name=test_name)
    #
    # manager.train(1000)
    # manager.test_td3_bc(corr_type=1, aug=2)
    #
    # model_name = "end_to_end"
    # test_name = "batch_aug"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, target=True, entire_trajectory=False, dimension=50000,
    #                   test_name=test_name)
    # manager.load(save_path)
    # manager.test_td3_bc(corr_type=1, aug=1)
    #
    # model_name = ""
    # test_name = "Noise"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, target=False, entire_trajectory=False, dimension=50000,
    #                   test_name=test_name)
    # manager.test_td3_bc(corr_type=1, aug=3)
    #
    # model_name = ""
    # test_name = "Vanilla"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                   writer_name=writer_name, test_aug=False, target=False, entire_trajectory=False, dimension=20000,
    #                   test_name=test_name)
    # manager.test_td3_bc(1)

