from torch import save

from Manager import Manager

if __name__ == "__main__":
    # env_name = "bullet-halfcheetah-medium-expert-v0"
    #
    # writer_name = "td3aug_medium_expert_bullet_10k"
    # save_path = "td3aug_medium_expert_bullet_models_10k"
    #

    # model_name = "end_to_end"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
    #                    writer_name=writer_name, test_aug=False, test_aug_dimension=50000, target=True, entire_trajectory=False)
    #
    # manager.train_nd_test(1000, 1, aug_td3=True)

    # model_name = "end_to_end"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
    #                    writer_name=writer_name, test_aug=False, test_aug_dimension=50000, target=True, entire_trajectory=False)
    #
    # manager.train_nd_test(1000, 1, aug_td3=True)

    # model_name = "No_corruption"  # "end_to_end" "splitted"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
    #                  writer_name=writer_name, test_aug=True, test_aug_dimension=50000, entire_trajectory=False)
    # manager.test_td3_bc(1)

    #########################

    env_name = "halfcheetah-medium-replay-v2"

    writer_name = "td3aug_medium_expert_mujoco_100k"
    save_path = "Models/td3aug_medium_expert_mujoco_models_100k"

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
                       writer_name=writer_name, test_aug=False,  target=True, entire_trajectory=False)
    # manager.load(save_path)
    manager.train_nd_test(1000, 0, aug_td3=False)
    # manager.load(save_path)
    # manager.test_VAE()
    # manager.test_td3_bc(0)


    # model_name = "No_corruption"  # "end_to_end" "splitted"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
    #                 writer_name=writer_name, test_aug=True,  entire_trajectory=False)
    # manager.test_td3_bc(1)
    #
    model_name = "Removed"  # "end_to_end" "splitted"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
                    writer_name=writer_name, test_aug=False,  entire_trajectory=False)
    manager.test_td3_bc(4)

