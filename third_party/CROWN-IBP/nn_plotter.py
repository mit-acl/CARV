import matplotlib.pyplot as plt
import pickle
import torch

from model_defs import model_mlp_any
from bound_layers import BoundSequential

def main():
    
    file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator/xs.pkl', 'rb')
    xs = pickle.load(file)
    file.close()

    file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator/us.pkl', 'rb')
    us_true = pickle.load(file)
    file.close()

    neurons = [15, 10, 5]
    model = model_mlp_any(2, neurons, 1)
    bound_model = BoundSequential.convert(model)


    model_file = './double_integrator_crown/double_integrator_default_best.pth'
    #model_file += "_pretrain"
    print("Loading model file", model_file)
    checkpoint = torch.load(model_file)
    if isinstance(checkpoint["state_dict"], list):
        checkpoint["state_dict"] = checkpoint["state_dict"][0]
    new_state_dict = {}
    for k in checkpoint["state_dict"].keys():
        if "prev" in k:
            pass
        else:
            new_state_dict[k] = checkpoint["state_dict"][k]
    checkpoint["state_dict"] = new_state_dict
    
    """
    state_dict = m.state_dict()
    state_dict.update(checkpoint["state_dict"])
    m.load_state_dict(state_dict)
    print(checkpoint["state_dict"]["__mask_layer.weight"])
    """

    model.load_state_dict(checkpoint["state_dict"])

    import pdb; pdb.set_trace()

    us_model = model(torch.tensor(xs).float()).detach().numpy()

    # model.load_state_dict(torch.load('./double_integrator_crown/double_integrator_default_best.pth'))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(xs[:, 0], xs[:, 1], us_true[:, 0], c = 'b', marker='o')
    ax.scatter(xs[:, 0], xs[:, 1], us_model[:, 0], c = 'r', marker='o')
    plt.show()


if __name__ == "__main__":
    main()