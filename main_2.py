from Manager import Manager

if __name__ == "__main__":
    env_name = "bullet-halfcheetah-medium-replay-v0"

    writer_name = "test_medium_replay_pybullet"
    save_path = "replay_pybullet_models"

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
                      writer_name=writer_name)
    # manager.train_nd_test(500, 0)
    manager.load(save_path)
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
