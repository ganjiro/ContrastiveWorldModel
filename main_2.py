from Manager import Manager

if __name__ == "__main__":

    env_name = "halfcheetah-medium-expert-v2"

    writer_name = "Writers/td3aug_medium_expert_mujoco_50kRand_noNorm_fix"
    save_path = "Models/td3aug_medium_expert_mujoco_50kRand_noNorm_fix"


    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0,
                      writer_name=writer_name, test_aug=True, target=True, entire_trajectory=False, dimension=50000)

    manager.load(save_path)
    manager.test_VAE()
    manager.test_depth_mse()
    manager.test_world_model()
    manager.test_distr()

    # model_name = "No_corruption"  # "end_to_end" "splitted"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(1)
    #
    # model_name = "removed"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(4)
    #
    # model_name = "mean"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(2)
