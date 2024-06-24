import os

PATH = os.getcwd()

def get_plot_filename(params):

    if params["analysis"]["reachability_direction"] == "forward":
        examples_dir = "forward"
    else:
        examples_dir = "backward"
    save_dir = f"{PATH}/nfl_robustness_training/src/plots/{examples_dir}/"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = f'{save_dir}/{params["system"]["type"]}_{params["system"]["controller"]}_{params["system"]["feedback"]}_{params["analysis"]["partitioner"]["type"]}_{params["analysis"]["propagator"]["type"]}_tmax_{str(round(params["analysis"]["t_max"], 1))}_{params["analysis"]["propagator"]["boundary_type"]}'  # noqa: E501
    plot_filename += ".png"
    return plot_filename