from Manager import Manager

if __name__ == "__main__":
    env_name = "antmaze-medium-play-v2"

    writer_name = "Writers/antmaze_P_test_fill_gap/"
    save_path = "Models/antmaze_P_test_fill_gap"

    model_name = "end_to_end"
    test_name = "fill_gap"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0.7,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=989000,
                      test_name=test_name)

    #manager.train(200)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=0, aug=0)

    model_name = "end_to_end"
    test_name = "fill_gap"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=989000,
                      test_name=test_name)

    manager.test_td3_bc(corr_type=0, aug=0)

    model_name = ""
    test_name = "remove"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.7,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=989000,
                      test_name=test_name)

    # manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=0)


