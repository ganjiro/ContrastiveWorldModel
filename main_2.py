from Manager import Manager

if __name__ == "__main__":
    env_name = "halfcheetah-medium-v2"

    writer_name = "Writers/halfcheetah_M_2_test_aug_fulldataset/"
    save_path = "Models/halfcheetah_M_2_test_aug_fulldataset"

    model_name = "end_to_end"
    test_name = "OEB"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
                      test_name=test_name)

    manager.train(500)
    # manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=2)

    model_name = "end_to_end"
    test_name = "batch_aug"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=1)

    model_name = "end_to_end"
    test_name = "S4RL"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
                      test_name=test_name)
    manager.load(save_path)
    manager.test_td3_bc(corr_type=1, aug=4)

    model_name = ""
    test_name = "Noise"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
                      test_name=test_name)
    manager.test_td3_bc(corr_type=1, aug=3)

    model_name = ""
    test_name = "Vanilla"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0,
                      writer_name=writer_name, test_aug=False, entire_trajectory=True, dimension=None,
                      test_name=test_name)
    manager.test_td3_bc(1)
