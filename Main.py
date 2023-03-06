from Manager import Manager

if __name__ == "__main__":
    env_name = "bullet-halfcheetah-medium-expert-v0"

    writer_name = "Complete_test_05"
    save_path = "models_05"

    # model_name = "No_corruption"  # "end_to_end" "splitted"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(1)
    #
    # model_name = "removed"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(4)
    #
    # model_name = "mean"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(2)
    #
    # model_name = "noisy"
    # manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
    #                   writer_name=writer_name)
    # manager.test_td3_bc(3)

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
                      writer_name=writer_name)
    manager.train_nd_test(500, 0)

    model_name = "splitted"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=True, perc=0.5,
                      writer_name=writer_name)
    manager.train_nd_test(500, 0)

    model_name = "end_to_end"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0.5,
                      writer_name=writer_name)
    manager.train_nd_test(500, 0)

    model_name = "splitted"
    manager = Manager(model_name=model_name, env_name=env_name, savepath=save_path, contrastive=False, perc=0.5,
                      writer_name=writer_name)
    manager.train_nd_test(500, 0)




