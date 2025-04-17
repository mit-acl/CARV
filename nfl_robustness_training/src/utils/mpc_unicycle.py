import numpy as np
import do_mpc

from casadi import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import yaml

# dir_path = os.getcwd()
dir_path = os.getcwd()+'/CARV'
print(f'dir_path = {dir_path}')

def setup_model(dt = 0.1, obstacles = []):
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    v = 1.

    pos = model.set_variable(var_type='_x', var_name='pos', shape=(2,1))
    psi = model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    # t = model.set_variable(var_type='_x', var_name='time', shape=(1,1))

    # v = model.set_variable(var_type='_u', var_name='v')
    omega = model.set_variable(var_type='_u', var_name='omega')

    pos_1 = vertcat(
        v*cos(psi),
        v*sin(psi)
    )
    model.set_rhs('pos', pos + dt * pos_1)
    model.set_rhs('psi', psi + dt * omega)
    # model.set_rhs('time', t + dt)

    obstacle_distance = []
    for obs in obstacles:
        d0 = sqrt((pos[0]-obs['x'])**2+(pos[1]-obs['y'])**2) - 1.*obs['r']
        obstacle_distance.extend([d0])

    # obs = obstacles[0]
    # d0 = (pos[0]-obs['x'])**2+(pos[1]-obs['y'])**2 - 1.01*obs['r']**2


    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))
    # model.set_expression('obstacle_distance', d0)
 
    model.setup()
    return model

def setup_mpc_controller(model, dt = 0.1):
    

    mpc = do_mpc.controller.MPC(model)

    goal = (2, 0)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 15,
        't_step': dt,
        'state_discretization': 'discrete',
        'store_full_solution':True,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'},
    }
    mpc.set_param(**setup_mpc)

    mterm = sqrt((model.x['pos'][0]-goal[0])**2 + (model.x['pos'][1]-goal[1])**2)
    lterm = sqrt((model.x['pos'][0]-goal[0])**2 + (model.x['pos'][1]-goal[1])**2)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(
        omega=1e-2
    )

    # Lower bounds on inputs:
    # mpc.bounds['lower','_u', 'v'] = 0
    mpc.bounds['lower','_u', 'omega'] = -0.3*np.pi
    # Lower bounds on inputs:
    # mpc.bounds['upper','_u', 'v'] = 2
    mpc.bounds['upper','_u', 'omega'] = 0.3*np.pi

    mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)

    mpc.setup()

    return mpc

def setup_simulator(model, dt = 0.1):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = dt)
    simulator.setup()

    return simulator

def setup_graphics(mpc, simulator):
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True

    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
    fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
        g.add_line(var_type='_x', var_name='pos', axis=ax[0])


        # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
        g.add_line(var_type='_u', var_name='phi_m_1_set', axis=ax[1])
        g.add_line(var_type='_u', var_name='phi_m_2_set', axis=ax[1])


    ax[0].set_ylabel('angle position [rad]')
    ax[1].set_ylabel('motor angle [rad]')
    ax[1].set_xlabel('time [s]')


def main(noisy = False):
    np.random.seed(0)
    dt = 0.2
    obstacles = [{'x': -5.5, 'y': 0., 'r': 2.},
                 {'x': -1.5, 'y': 1.5, 'r': 1.2}]
    model = setup_model(obstacles=obstacles, dt=dt)
    mpc = setup_mpc_controller(model, dt=dt)
    simulator = setup_simulator(model, dt=dt)


    x0 = np.array([-9, 3, -np.pi/6])
    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()

    for i in range(600):
        ptb = np.zeros((3, 1))
        if noisy:
            ptb = np.random.random((3, 1))*np.array([[0.25], [0.25], [0.125]])
        u0 = mpc.make_step(x0)

        # import pdb; pdb.set_trace()
        x0 = simulator.make_step(u0) + ptb
        
        # if np.linalg.norm(x0[:2], inf) < 1e-0:
        #         break

        if x0[0] > 0:
                break
        
    print("Path Completed in {} timesteps".format(i))

    xs = simulator.data['_x']
    fig, ax = plt.subplots()
    for obstacle in obstacles:
        circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue')
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')
    plt.show()

def read_obstacles(environment='safety_test', config_path=''):
    """
    Read circular obstacles for a yaml file for use with the simulator_lidar repository
    """
    with open(config_path+environment+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        # load environment area parameters
        area = yaml_data['area']
        area = (area['x_min'], area['x_max'], area['y_min'], area['y_max'])

        # load static and dynamic obstacles
        obs = yaml_data['obstacles']
        # all_obstacles = []
        # for i in range(len(obs)):
        #     obs_i = [{'x': obs[i]['centroid_x'], 'y': obs[i]['centroid_y'], 'r': obs[i]['radius']}]
        #     all_obstacles.append(obs_i)
        all_obstacles = [{'x': o['centroid_x'], 'y': o['centroid_y'], 'r': o['radius']} for o in obs]
        
    return all_obstacles

def generate_data(num_trajectories = 50, system = 'unicycle', noisy = False, prob_short = 0., use_config=False, environment='safety_test', config_path='', save_dir=None):
    dt = 0.2
    if not use_config:
        # obstacles = [{'x': -10, 'y': -1, 'r': 4},
        #              {'x': -3.5, 'y': 3, 'r': 3}]
        obstacles = [{'x': -6, 'y': -0.5, 'r': 2.75},
                    {'x': -1.25, 'y': 1.75, 'r': 2.}]
    else:
        obstacles = read_obstacles(environment=environment, config_path=config_path)

    model = setup_model(obstacles=obstacles, dt=dt)
    mpc = setup_mpc_controller(model, dt=dt)
    simulator = setup_simulator(model, dt=dt)


    num_states = 3
    # x0_range = np.array([
    #     [-18., -16.],
    #     [4., 6.],
    #     [-np.pi/4, np.pi/4]
    # ])
    x0_range = np.array([
        [-10., -9.],
        [3., 4.],
        [-np.pi/6, np.pi/6]
    ])

    x0s = np.random.uniform(
        low=x0_range[:, 0],
        high=x0_range[:, 1],
        size=(num_trajectories, num_states),
    )
    xs = None

    traj_num = 0
    info = {}

    for x0 in x0s:
        print("############################################################################################")
        print("#################################### Trajectory {} #########################################".format(traj_num))
        print("############################################################################################")
        print("############################################################################################")
        simulator.x0 = x0
        mpc.x0 = x0
        mpc.set_initial_guess()

        max_steps = 320
        short_traj_rand = np.random.random()
        if prob_short > short_traj_rand:
            max_steps = 15
        
        for i in range(max_steps):
            ptb = np.zeros((3, 1))
            if noisy:
                ptb_range = np.array([[0.5], [0.5], [np.pi/3]])
                ptb = np.random.random((3, 1))*ptb_range - ptb_range/2
            u0 = mpc.make_step(x0)
            x0 = simulator.make_step(u0) + ptb
            # if np.linalg.norm(x0[:2]) < 1e-1:
            #     break

            if x0[0] > 0:
                break
        
        if xs is None:
            xs = simulator.data['_x']
            us = simulator.data['_u']
        else:
            xs = np.vstack((xs, simulator.data['_x']))
            us = np.vstack((us, simulator.data['_u']))
        
        info[traj_num] = {'_time': simulator.data['_time'], '_x': simulator.data['_x'], '_u': simulator.data['_u']}
        traj_num += 1

        mpc.reset_history()
        simulator.reset_history()


    dataset_name = "double_obstacle_aug"
    path = "{}/nfl_robustness_training/src/_static/datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path + "/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)

    with open(path + "/info.pkl", "wb") as f:
        pickle.dump([xs, us], f)

    fig, ax = plt.subplots()
    for obstacle in obstacles:
        circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue', zorder = 0)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')
    plt.show()
    print(f'data saved to {path}')

def correct_data():
    system = "unicycle"
    dataset_name = "double_obstacle"
    # path = "{}/nfl_robustness_training/src/_static/datasets/{}/{}".format(dir_path, system, dataset_name)
    with open(dir_path + "/nfl_robustness_training/src/_static/datasets/unicycle/" + dataset_name + "/dataset.pkl", 'rb') as f:
        xs, us = pickle.load(f)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # main(noisy=False)
    generate_data(num_trajectories=1, noisy=False, prob_short=0.0, use_config=True, config_path='/Users/dgaillard/Desktop/aerospace_controls_lab/learned_control/simulator_lidar/config/')
    # correct_data()
    
