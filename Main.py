from Manager import Manager


if __name__ == "__main__":

    env_name = "bullet-halfcheetah-medium-expert-v0"

    model_name = "end_to_end"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=True)

    manager.load("ManagerSave")

    manager.test_td3_bc(0.3, 0)

    model_name = "splitted"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=True)

    manager.load("ManagerSave")

    manager.test_td3_bc(0.3, 0)

    model_name = "end_to_end"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=False)

    manager.load("ManagerSave")

    manager.test_td3_bc(0.3, 0)

    model_name = "splitted"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=False)

    manager.load("ManagerSave")

    manager.test_td3_bc(0.3, 0)

    model_name = "mean" # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=True)

    #manager.train(500)

    #manager.load("ManagerSave")

    #manager.test_render()

    manager.test_td3_bc(0.3, 2)

    model_name = "noisy"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=True)

    # manager.train(500)

    # manager.load("ManagerSave")

    # manager.test_render()

    manager.test_td3_bc(0.3, 3)

    model_name = "removed"  # "end_to_end" # "splitted"

    manager = Manager(model_name, env_name, "ManagerSave", contrastive=True)

    # manager.train(500)

    # manager.load("ManagerSave")

    # manager.test_render()

    manager.test_td3_bc(0.3, 4)



