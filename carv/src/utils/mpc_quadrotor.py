import numpy as np
import do_mpc

from casadi import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os

dir_path = os.getcwd()

def setup_model(dt = 0.1, obstacles = []):
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    normal = True
    g = 9.81

    pos = model.set_variable(var_type='_x', var_name='pos', shape=(3,1))
    vel = model.set_variable(var_type='_x', var_name='vel', shape=(3,1))
    if normal:
        roll = model.set_variable(var_type='_u', var_name='roll')
        pitch = model.set_variable(var_type='_u', var_name='pitch')
        thrust = model.set_variable(var_type='_u', var_name='thrust')
    else:
        slider = model.set_variable(var_type='_u', var_name='slider')
        magnitude = model.set_variable(var_type='_u', var_name='magnitude')
        thrust = model.set_variable(var_type='_u', var_name='thrust')

        roll = (1-slider) * magnitude
        pitch = (slider) * magnitude

    pos_1 = pos + dt * vel
    vel_1 = vel + dt * vertcat(g*tan(roll), -g*tan(pitch), thrust - g)


    model.set_rhs('pos', pos_1)
    model.set_rhs('vel', vel_1)

    obstacle_distance = []
    for obs in obstacles:
        if 'z' in obs and 'y' in obs:
            d0 = sqrt((pos[0]-obs['x'])**2+(pos[1]-obs['y'])**2+(pos[2]-obs['z'])**2) - 1.*obs['r']
        elif 'y' in obs:
            d0 = sqrt((pos[0]-obs['x'])**2+(pos[1]-obs['y'])**2) - 1.*obs['r']
        elif 'z' in obs:
            d0 = sqrt((pos[0]-obs['x'])**2+(pos[2]-obs['z'])**2) - 1.*obs['r']
        obstacle_distance.extend([d0])

    # obs = obstacles[0]
    # d0 = (pos[0]-obs['x'])**2+(pos[1]-obs['y'])**2 - 1.01*obs['r']**2


    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))
    # model.set_expression('obstacle_distance', d0)
 
    model.setup()
    return model

def setup_mpc_controller(model, dt = 0.1):
    

    mpc = do_mpc.controller.MPC(model)

    goal = (20, 0, 1)
    normal = True
    g = 9.81

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 60,
        't_step': dt,
        'state_discretization': 'discrete',
        'store_full_solution':True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'},
    }
    mpc.set_param(**setup_mpc)
    mpc.u0['thrust', 0] = g

    mterm = sqrt((model.x['pos'][0]-goal[0])**2 + (model.x['pos'][1]-goal[1])**2 + (model.x['pos'][2]-goal[2])**2)
    lterm = sqrt((model.x['pos'][0]-goal[0])**2 + (model.x['pos'][1]-goal[1])**2 + (model.x['pos'][2]-goal[2])**2)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    if normal:
        mpc.set_rterm(
            roll=5e1,
            pitch=5e1,
            thrust=1e-1
        )

        # Bounds on inputs:
        mpc.bounds['lower', '_u', 'roll'] = -np.pi/6
        mpc.bounds['upper', '_u', 'roll'] = np.pi/6
        mpc.bounds['lower', '_u', 'pitch'] = -np.pi/6
        mpc.bounds['upper', '_u', 'pitch'] = np.pi/6
    
    else:
        mpc.set_rterm(
            slider=1e-2,
            magnitude=1e-1,
            thrust=1e-2
        )
        mpc.bounds['lower', '_u', 'slider'] = 0
        mpc.bounds['upper', '_u', 'slider'] = 1
        mpc.bounds['lower', '_u', 'magnitude'] = -np.pi/6
        mpc.bounds['upper', '_u', 'magnitude'] = np.pi/6

    mpc.bounds['lower', '_u', 'thrust'] = g - 0.1*g
    mpc.bounds['upper', '_u', 'thrust'] = g + 0.1*g

    # Bounds on states:
    mpc.bounds['lower', '_x', 'vel'] = -1
    mpc.bounds['upper', '_x', 'vel'] = 1

    # Lower bounds on inputs:
    # mpc.bounds['lower','_u', 'v'] = 0
    # mpc.bounds['lower','_u', 'omega'] = -0.3*np.pi
    # mpc.bounds['upper','_u', ''] = 2
    # mpc.bounds['upper','_u', 'omega'] = 0.3*np.pi

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
    dimensionality = 3
    # obstacles = [{'x': -5.5, 'y': 0., 'r': 2.},
    #              {'x': -1.5, 'y': 2., 'r': 2}]
    # obstacles = [{'x': -5.5, 'y': -2., 'r': 2.},
    #              {'x': -1.5, 'y': 4., 'r': 2}]
    if dimensionality == 2:
        obstacles = [{'x': -6, 'y': -0.5, 'r': 2.75},
                    {'x': -9, 'y': -0.5, 'r': 2.75},
                    {'x': -1.25, 'y': 1.75, 'r': 2.},
                    {'x': -1.25, 'y': 4.75, 'r': 2.},
                    {'x': -1.25, 'y': 7.75, 'r': 2.6},
                    {'x': -1.25, 'y': 3.25, 'r': 2.6},
                    {'x': -1.25, 'y': 6.25, 'r': 2.6},
                    {'x': -1.25, 'y': 9.25, 'r': 2.6}]
        x0 = np.array([-10, 3, 1, 1, 0, 0])
    else:
        yoffset1 = 2
        zoffset1 = 2
        yoffset2 = -2
        zoffset2 = 2
        little_radius = 1.25
        obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': -4. + yoffset1, 'r': 2.},
                     {'x': -6, 'y': 4. + yoffset1, 'r': 2.},
                     {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': -3. + zoffset1, 'r': 2.},
                     {'x': -6, 'z': 5. + zoffset1, 'r': 2.},
        
                     {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': -4. + yoffset2, 'r': 2.},
                     {'x': -3, 'y': 4. + yoffset2, 'r': 2.},
                     {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': -3. + zoffset2, 'r': 2.},
                     {'x': -3, 'z': 5. + zoffset2, 'r': 2.},]
        x0 = np.array([-10, 0, 1, 1, 0, 0])
        
    model = setup_model(obstacles=obstacles, dt=dt)
    mpc = setup_mpc_controller(model, dt=dt)
    simulator = setup_simulator(model, dt=dt)


    
    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()

    for i in range(600):
        ptb = np.zeros((6, 1))
        if noisy:
            ptb = np.random.random((3, 1))*np.array([[0.25], [0.25], [0.25], [0.125], [0.125], [0.125]])
        u0 = mpc.make_step(x0)

        # import pdb; pdb.set_trace()
        x0 = simulator.make_step(u0) + ptb
        
        # if np.linalg.norm(x0[:2], inf) < 1e-0:
        #         break

        if x0[0] > 0:
                break
        
    print("Path Completed in {} timesteps".format(i))

    xs = simulator.data['_x']
    if dimensionality == 2:
        fig, ax = plt.subplots()
        for obstacle in obstacles:
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue')
            ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for obstacle in obstacles:
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = obstacle['r'] * np.cos(v) + obstacle['z']
            elif 'y' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                z = np.outer(np.ones(np.size(u)), v)

            elif 'z' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
                y = np.outer(np.ones(np.size(u)), v)
            
            ax.plot_surface(x, y, z, color='b', alpha=0.1)
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], 'k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal')
    plt.show()

def generate_data(num_trajectories = 50, system = 'quadrotor', noisy = False, prob_short = 0.):
    dt = 0.2
    # obstacles = [{'x': -10, 'y': -1, 'r': 4},
    #              {'x': -3.5, 'y': 3, 'r': 3}]
    # obstacles = [{'x': -6, 'y': -0.5, 'r': 2.75},
    #              {'x': -1.25, 'y': 1.75, 'r': 2.}]
    # obstacles = [{'x': -6, 'y': -0.5, 'r': 2.75},
    #              {'x': -9, 'y': -0.5, 'r': 2.75},
    #              {'x': -1.25, 'y': 1.75, 'r': 2.},
    #              {'x': -1.25, 'y': 4.75, 'r': 2.},
    #              {'x': -1.25, 'y': 7.75, 'r': 2.6},
    #              {'x': -1.25, 'y': 3.25, 'r': 2.6},
    #              {'x': -1.25, 'y': 6.25, 'r': 2.6},
    #              {'x': -1.25, 'y': 9.25, 'r': 2.6}]

    dimensionality = 3
    # obstacles = [{'x': -5.5, 'y': 0., 'r': 2.},
    #              {'x': -1.5, 'y': 2., 'r': 2}]
    # obstacles = [{'x': -5.5, 'y': -2., 'r': 2.},
    #              {'x': -1.5, 'y': 4., 'r': 2}]
    if dimensionality == 2:
        obstacles = [{'x': -6, 'y': -0.5, 'r': 2.75},
                    {'x': -9, 'y': -0.5, 'r': 2.75},
                    {'x': -1.25, 'y': 1.75, 'r': 2.},
                    {'x': -1.25, 'y': 4.75, 'r': 2.},
                    {'x': -1.25, 'y': 7.75, 'r': 2.6},
                    {'x': -1.25, 'y': 3.25, 'r': 2.6},
                    {'x': -1.25, 'y': 6.25, 'r': 2.6},
                    {'x': -1.25, 'y': 9.25, 'r': 2.6}]
        # x0 = np.array([-10, 3, 1, 1, 0, 0])
        x0_range = np.array([
            [-11, -10],
            [2.5, 3.5],
            [0.75, 1.22],
            [0.75, 1.25],
            [-0.25, 0.25],
            [-0.25, 0.25],
        ])
    else:
        yoffset1 = 1
        zoffset1 = 3
        yoffset2 = -1.5
        zoffset2 = 1
        little_radius = 1.25
        big_radius = 3
        obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
                     {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
        
                     {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
                     {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},]
        # x0 = np.array([-10, 0, 1, 1, 0, 0])
        x0_range = np.array([
            [-10.5, -9.5],
            [-0.5, 0.5],
            [0.75, 1.25],
            [0.75, 1.25],
            [-0.25, 0.25],
            [-0.25, 0.25],
        ])
    
    model = setup_model(obstacles=obstacles, dt=dt)
    mpc = setup_mpc_controller(model, dt=dt)
    simulator = setup_simulator(model, dt=dt)


    num_states = 6
    

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
            ptb = np.zeros((6, 1))
            if noisy:
                ptb_range = np.array([[0.05], [0.05], [0.05], [0.05], [0.05], [0.05]])
                ptb = np.random.random((6, 1))*ptb_range - ptb_range/2
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


    dataset_name = "gates"
    path = "{}/nfl_robustness_training/src/_static/datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path + "/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)

    with open(path + "/info.pkl", "wb") as f:
        pickle.dump([xs, us], f)

    print(len(xs))

    # fig, ax = plt.subplots()
    # for obstacle in obstacles:
    #     circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue', zorder = 0)
    #     ax.add_patch(circle)
    # ax.set_aspect('equal')
    # ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')
    if dimensionality == 2:
        fig, ax = plt.subplots()
        for obstacle in obstacles:
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue')
            ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for obstacle in obstacles:
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = obstacle['r'] * np.cos(v) + obstacle['z']
            elif 'y' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                z = np.outer(np.ones(np.size(u)), v)

            elif 'z' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
                y = np.outer(np.ones(np.size(u)), v)
            ax.plot_surface(x, y, z, color='b', alpha=0.1)
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], 'k.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.show()

def correct_data():
    system = "quadrotor"
    dataset_name = "three_dimension"
    # path = "{}/nfl_robustness_training/src/_static/datasets/{}/{}".format(dir_path, system, dataset_name)
    with open(dir_path + "/nfl_robustness_training/src/_static/datasets/quadrotor/" + dataset_name + "/dataset.pkl", 'rb') as f:
        xs, us = pickle.load(f)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    # main(noisy=False)
    generate_data(num_trajectories=1000, noisy=True, prob_short=0.0)
    # correct_data()    
