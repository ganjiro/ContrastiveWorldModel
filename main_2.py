from Manager import Manager

if __name__ == "__main__":
    env_name = "halfcheetah-medium-expert-v2"

    writer_name = "test_aug/test_100k"
    # save_path = "models_mediumreplay_05"

    # model_name = "No_augmentation"  # "end_to_end" "splitted"
    # manager = Manager(model_name=model_name, env_name=env_name, contrastive=True, writer_name=writer_name, test_aug=True)
    # manager.test_td3_bc(1)

    # model_name = "end_to_end"
    # manager = Manager(model_name=model_name, env_name=env_name, contrastive=True, writer_name=writer_name,
    #                   test_aug=True)
    # manager.train_nd_test(500, 0)

    # model_name = "end_to_end"
    # manager = Manager(model_name=model_name, env_name=env_name, contrastive=False, writer_name=writer_name,
    #                   test_aug=True)
    # manager.train_nd_test(500, 0)

    ###############################

    writer_name = "test_aug/test_200k"
    # save_path = "models_mediumreplay_05"

    model_name = "No_augmentation"  # "end_to_end" "splitted"
    manager = Manager(model_name=model_name, env_name=env_name, contrastive=True, writer_name=writer_name,
                      test_aug=True, test_aug_dimension=200000)
    manager.test_td3_bc(1)

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, contrastive=True, writer_name=writer_name,
                      test_aug=True, test_aug_dimension=200000)
    manager.train_nd_test(500, 0)

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, contrastive=False, writer_name=writer_name,
                      test_aug=True, test_aug_dimension=200000)
    manager.train_nd_test(500, 0)
