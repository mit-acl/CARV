import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle
import os

import nfl_veripy.dynamics as dynamics  # noqa: E402
from utils.nn import *

dir_path = os.path.dirname(os.path.realpath(__file__))

# nicks colorblind color palette (bad)
lightblue = '#56B4E9'
lightorange = '#FFC20A'
blue = '#2E72F2' # '#5087F5' # '#005AB5'# '#1C44FE'
green = '#28DD51'
orange = '#FB762F' # '#D11F40' # '#D66A37' # '#DC3220'
magenta = '#FB5CDB'

# Wong colorblind palette
# lightblue = '#56B4E9'
# lightorange = '#E69F00'
# blue = '#0072B2' # '#5087F5' # '#005AB5'# 
# green = '#009E73'
# orange = '#D55E00' # '#D11F40' # '#D66A37' # 
# magenta = '#CC79A7'

# lightblue = '#56B4E9'
# lightorange = '#FFC20A'
# blue = '#56B4E9' # '#5087F5' # '#005AB5'# '#1C44FE'
# green = '#56B4E9'
# orange = '#D55E00' # '#D11F40' # '#D66A37' # '#DC3220'
# magenta = '#FB5CDB'

def official_plotter(info, cl_system, frames = [-1], save_animation=False, save=False, name='default'):
    num_steps = len(info[-1]["reachable_sets"]) - 1
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 20,
    })

    reachable_set_snapshots = []
    # info[-1]['time'] = 3
    for j, snapshot in enumerate(info):
        num_times = 1
        if j == len(info) - 1:
            num_times = 20
        reachable_set_snapshot = []
        for i, reach_set_tuple in enumerate(snapshot['reachable_sets']):
            state_range = reach_set_tuple[0]
            is_symbolic = reach_set_tuple[1]
            collides = reach_set_tuple[2]
            edgecolor = blue # '#2176FF'

            if i == snapshot['child_idx'] and snapshot['child_idx'] != snapshot['parent_idx'] + 1:
                edgecolor = blue #'#D63230' #F45B69'
            
            if i == 0:
                edgecolor = 'k'
            elif is_symbolic:
                edgecolor = green # '#00CC00' # edgecolor = '#D112E2' # '#D14081' # '#53917E' # '#20A39E' # '#44BBA4'

            if i == snapshot['parent_idx'] and j < len(info) - 1:
                edgecolor = magenta #'#FF00FF' # '#FFAE03'
            if collides:
                edgecolor = orange # '#FF8000' # '#D63230'
            
            
            reachable_set_snapshot.append((state_range, edgecolor))
        
        for i in range(num_times):
            reachable_set_snapshots.append(reachable_set_snapshot)
            

    def animate(i):
        ax.clear()
        state_range = info[0]["reachable_sets"][0][0]
        num_trajectories = 500
        xs = sample_from_range(state_range, cl_system, num_steps, num_trajectories = num_trajectories)
        
        plot_exact_reachable_sets = False
        if plot_exact_reachable_sets:
            xs_sorted = xs.reshape((num_steps+1, num_trajectories, 3))
            for timestep_states in xs_sorted:
                xy = np.min(timestep_states, axis=0)
                width, height = (np.max(timestep_states, axis=0) - xy)[[0, 1]]
                rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
                ax.add_patch(rect)

        
        ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

        # xy = initial_set[[0, 1], 0]
        # width, height = initial_set[[0, 1], 1] - initial_set[[0, 1], 0]
        # rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect)
        
        if cl_system.dynamics.name == "DoubleIntegrator":
            fs = 26
            plt.rcParams.update({"font.size": fs})
            plt.subplots_adjust(left=0.176, bottom=0.15, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
            fig.set_size_inches(9.6, 7.2)
            constraint_color = '#262626'
            ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c=constraint_color, linewidth=2)
            rect = Rectangle(np.array([0.0, -1.25]), 3.25, 0.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
            ax.add_patch(rect)

            ax.plot(np.array([0, 0]), np.array([-1.25, 1.]), c=constraint_color, linewidth=2)
            rect = Rectangle(np.array([-1.5, -1.25]), 1.5, 2.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
            ax.add_patch(rect)

            ax.set_xlim([-0.5, 3.25])
            ax.set_ylim([-1.25, 0.5])

            ax.set_xlabel('$\mathbf{x}[0]$', fontsize=fs)
            ax.set_ylabel('$\mathbf{x}[1]$', fontsize=fs)

            linewidth = 1.5

            
        elif cl_system.dynamics.name == "Unicycle_NL":
            fig.set_size_inches(10, 5)
            delta = 0.29
            # obstacles = [{'x': -10, 'y': -1, 'r': 3},
            #              {'x': -3, 'y': 2.5, 'r': 2 }]
            obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
                        {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
            for obstacle in obstacles:
                color = '#262626'
                circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor='none')
                ax.add_patch(circle)
                circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
                ax.add_patch(circle)

            ax.set_xlim([-10, 1])
            ax.set_ylim([-1, 4])
            ax.set_aspect('equal')

            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$y$ [m]')

            linewidth = 1

            
        

        

        for reachable_set_snapshot in reachable_set_snapshots[i]:
            set_range = reachable_set_snapshot[0]
            edgecolor = reachable_set_snapshot[1]
            xy = set_range[[0, 1], 0]
            width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
            rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)
            alpha = 0.2
            if edgecolor == '#FF8000':
                alpha = 0.4
            if edgecolor == orange:
                alpha = 0.4

            if edgecolor != 'k':
                rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=edgecolor, alpha=alpha)
                ax.add_patch(rect)

        
        # for j in range(i):
        #     set_range = reachable_sets_extended[j]
        #     xy = set_range[[0, 1], 0]
        #     width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
        #     rect = Rectangle(xy, width, height, linewidth=1, edgecolor=colors_extended[j], facecolor='none')



        #     # import pdb; pdb.set_trace()
        #     if j == i - 1 or not remove_extended[j]:
        #         ax.add_patch(rect)
    figure_path = dir_path + '/plots/forward/LCSS24/' + cl_system.dynamics.name + '/'
    if save_animation:
        ani_name = figure_path + cl_system.dynamics.name + '_' + name + '.gif'
        time_multiplier = 5
        ani = FuncAnimation(fig, animate, frames=len(reachable_set_snapshots), repeat=True)
        ani.save(ani_name, dpi=300, writer=PillowWriter(fps=time_multiplier*2))
    else:
        for i in frames:
            animate(i)
            plt.show()


def official_3D_plotter(info, cl_system, frames = [-1], save_animation=False, save=False, name='default'):
    num_steps = len(info[-1]["reachable_sets"]) - 1
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 20,
    })

    reachable_set_snapshots = []
    # info[-1]['time'] = 3
    for j, snapshot in enumerate(info):
        num_times = 1
        if j == len(info) - 1:
            num_times = 30
        reachable_set_snapshot = []
        for i, reach_set_tuple in enumerate(snapshot['reachable_sets']):
            state_range = reach_set_tuple[0]
            is_symbolic = reach_set_tuple[1]
            collides = reach_set_tuple[2]
            edgecolor = blue # '#2176FF'

            if i == snapshot['child_idx'] and snapshot['child_idx'] != snapshot['parent_idx'] + 1:
                edgecolor = blue #'#D63230' #F45B69'
            
            if i == 0:
                edgecolor = 'k'
            elif is_symbolic:
                edgecolor = green # '#00CC00' # edgecolor = '#D112E2' # '#D14081' # '#53917E' # '#20A39E' # '#44BBA4'

            if i == snapshot['parent_idx'] and j < len(info) - 1:
                edgecolor = magenta #'#FF00FF' # '#FFAE03'
            if collides:
                edgecolor = orange # '#FF8000' # '#D63230'
            
            
            reachable_set_snapshot.append((state_range, edgecolor))
        
        for i in range(num_times):
            reachable_set_snapshots.append(reachable_set_snapshot)
            

    def animate(i):    
        ax.clear()
        state_range = info[0]["reachable_sets"][0][0]
        num_trajectories = 100
        xs = sample_from_range(state_range, cl_system, num_steps, num_trajectories = num_trajectories)
        
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], s=1, c='k')
    

        fig.set_size_inches(10, 9)
        # yoffset1 = 2
        # zoffset1 = 2
        # yoffset2 = -1.5
        # zoffset2 = 0.25
        # little_radius = 1.25 * 0.5
        # big_radius = 2.5
        yoffset1 = 1
        zoffset1 = 3
        yoffset2 = -1.5
        zoffset2 = 1
        little_radius = 1.25*0.4
        big_radius = 0.3
        yoffset3 = 0
        zoffset3 = 0
        obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius, 'gate_number': 1},
                    {'x': -6, 'y': 2. + yoffset1, 'r': little_radius, 'gate_number': 1},
                    {'x': -6, 'z': -1. + zoffset1, 'r': little_radius, 'gate_number': 1},
                    {'x': -6, 'z': 3. + zoffset1, 'r': little_radius, 'gate_number': 1},

                    {'x': -3, 'y': -2. + yoffset2, 'r': little_radius, 'gate_number': 2},
                    {'x': -3, 'y': 2. + yoffset2, 'r': little_radius, 'gate_number': 2},
                    {'x': -3, 'z': -1. + zoffset2, 'r': little_radius, 'gate_number': 2},
                    {'x': -3, 'z': 3. + zoffset2, 'r': little_radius, 'gate_number': 2},
                    
                    {'x': 0, 'y': -2. + yoffset3, 'r': little_radius, 'gate_number': 3},
                    {'x': 0, 'y': 2. + yoffset3, 'r': little_radius, 'gate_number': 3},
                    {'x': 0, 'z': -1. + zoffset3, 'r': little_radius, 'gate_number': 3},
                    {'x': 0, 'z': 3. + zoffset3, 'r': little_radius, 'gate_number': 3},]
        # obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
        #             {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
        #             {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
        #             {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
        #             {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
        #             {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
        #             {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
        #             {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
        
        #             {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
        #             {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
        #             {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
        #             {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
        #             {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
        #             {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
        #             {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
        #             {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},
                    
        #             {'x': 0, 'y': -2. + yoffset3, 'r': little_radius},
        #             {'x': 0, 'y': 2. + yoffset3, 'r': little_radius},
        #             {'x': 0, 'z': -1. + zoffset3, 'r': little_radius},
        #             {'x': 0, 'z': 3. + zoffset3, 'r': little_radius},]
        for obstacle in obstacles:
            color = '#262626'
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = obstacle['r'] * np.cos(v) + obstacle['z']
            elif 'y' in obstacle and obstacle['r'] == little_radius:
                if obstacle['gate_number'] == 1:
                    offset = zoffset1 + 1
                elif obstacle['gate_number'] == 2:
                    offset = zoffset2 + 1
                elif obstacle['gate_number'] == 3:
                    offset = zoffset3 + 1

                height = (4 + little_radius)/2
                num_points = 64
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height + offset, height + offset, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                z = np.outer(np.ones(np.size(u)), v)

            elif 'z' in obstacle and obstacle['r'] == little_radius:
                if obstacle['gate_number'] == 1:
                    offset = yoffset1
                elif obstacle['gate_number'] == 2:
                    offset = yoffset2
                elif obstacle['gate_number'] == 3:
                    offset = yoffset3
                
                height = (4 + little_radius)/2
                num_points = 64
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height + offset, height + offset, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
                y = np.outer(np.ones(np.size(u)), v)
            ax.plot_surface(x, y, z, color=color, alpha=0.1)

        for reachable_set_snapshot in reachable_set_snapshots[i]:
            set_range = reachable_set_snapshot[0]
            edgecolor = reachable_set_snapshot[1]
            
            x = set_range[0, 0]
            y = set_range[1, 0]
            z = set_range[2, 0]
            dx = set_range[0, 1] - set_range[0, 0]
            dy = set_range[1, 1] - set_range[1, 0]
            dz = set_range[2, 1] - set_range[2, 0]
            # alpha = 0.18
            alpha = 0.15
            if edgecolor == green or edgecolor == magenta:
                alpha = 0.65
            if edgecolor == orange:
                alpha = 0.3
            ax.bar3d(x, y, z, dx, dy, dz, color=edgecolor, alpha=alpha)


        # 2D Projection
        zmin = -3

        ax.scatter(xs[:, 0], xs[:, 1], zmin, s=1, c='k', alpha=0.1)
        for obstacle in obstacles:
            color = '#262626'
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = zmin
            elif 'y' in obstacle and obstacle['r'] == little_radius:
                offset = zmin

                height = 0.01
                num_points = 64
                # u = np.linspace(0, 2 * np.pi, num_points)
                # v = np.linspace(-height + offset, height + offset, 2)
                # x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                # y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                # z = np.outer(np.ones(np.size(u)), v)
                # theta = np.linspace(0, 2 * np.pi, 100)
                # x = obstacle['x'] + obstacle['r'] * np.cos(theta)
                # y = obstacle['y'] + obstacle['r'] * np.sin(theta)
                # z = np.full_like(x, zmin)

                # import mpl_toolkits.mplot3d.art3d as
                import mpl_toolkits.mplot3d.art3d as art3d
                circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
                ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=zmin, zdir="z")

        for reachable_set_snapshot in reachable_set_snapshots[i]:
            set_range = reachable_set_snapshot[0]
            edgecolor = reachable_set_snapshot[1]
            
            x = set_range[0, 0]
            y = set_range[1, 0]
            z = zmin
            dx = set_range[0, 1] - set_range[0, 0]
            dy = set_range[1, 1] - set_range[1, 0]
            dz = 0.01
            # alpha = 0.15
            alpha = 0.18
            if edgecolor == green or edgecolor == magenta:
                alpha = 0.65
            if edgecolor == orange:
                alpha = 0.3
            ax.bar3d(x, y, z, dx, dy, dz, color=edgecolor, alpha=alpha)


        
        ymax = 5

        # ax.scatter(xs[:, 0], ymax, xs[:, 2], s=1, c='k', alpha=0.1)
        # for obstacle in obstacles:
        #     color = '#262626'
        #     if 'z' in obstacle and 'y' in obstacle:
        #         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #         x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
        #         y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
        #         z = zmin
        #     elif 'z' in obstacle and obstacle['r'] == little_radius:
        #         offset = ymax

        #         height = 0.01
        #         num_points = 64
        #         # u = np.linspace(0, 2 * np.pi, num_points)
        #         # v = np.linspace(-height + offset, height + offset, 2)
        #         # x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
        #         # y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
        #         # z = np.outer(np.ones(np.size(u)), v)
        #         # theta = np.linspace(0, 2 * np.pi, 100)
        #         # x = obstacle['x'] + obstacle['r'] * np.cos(theta)
        #         # y = obstacle['y'] + obstacle['r'] * np.sin(theta)
        #         # z = np.full_like(x, zmin)

        #         # import mpl_toolkits.mplot3d.art3d as
        #         import mpl_toolkits.mplot3d.art3d as art3d
        #         circle = plt.Circle((obstacle['x'], obstacle['z']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
        #         ax.add_patch(circle)
        #         art3d.pathpatch_2d_to_3d(circle, z=ymax, zdir="y")

        # for reachable_set_snapshot in reachable_set_snapshots[i]:
        #     set_range = reachable_set_snapshot[0]
        #     edgecolor = reachable_set_snapshot[1]
            
        #     x = set_range[0, 0]
        #     y = ymax
        #     z = set_range[2, 0]
        #     dx = set_range[0, 1] - set_range[0, 0]
        #     dy = 0.01
        #     dz = set_range[2, 1] - set_range[2, 0]
        #     alpha = 0.2
        #     if edgecolor == '#FF8000': 
        #         alpha = 0.6
        #     ax.bar3d(x, y, z, dx, dy, dz, color=edgecolor, alpha=alpha)
        
        

        ax.set_xlim([-11.5, 2])
        ax.set_ylim([-5, ymax])
        # ax.set_zlim([zmin, 7])
        ax.set_zlim([zmin, 6])
        ax.set_aspect('equal')
        ax.set_zticks(np.arange(-2, 7, 2))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # elevation = 16
        # azimuth = -80
        elevation = 22
        azimuth = -99
        if save_animation:
            azimuth = -105 + np.sin(np.pi/2*i/len(reachable_set_snapshots)) * 20 + 20
            elevation = 22 - np.sin(np.pi/2*i/len(reachable_set_snapshots)) * 6
            # azimuth = -105 - np.cos(np.pi*i/len(reachable_set_snapshots)) * 10 + 10
            # elevation = 22 - np.sin(np.pi/2*i/len(reachable_set_snapshots)) * 6
        ax.view_init(elevation, azimuth)

        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        ax.set_xlabel('$p_x$ [m]')
        ax.set_ylabel('$p_y$ [m]')
        ax.set_zlabel('$p_z$ [m]')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

    figure_path = dir_path + '/plots/forward/LCSS24/' + cl_system.dynamics.name + '/'
    if save_animation:
        ani_name = figure_path + cl_system.dynamics.name + '_' + name + '.gif'
        time_multiplier = 5
        ani = FuncAnimation(fig, animate, frames=len(reachable_set_snapshots), repeat=True)
        ani.save(ani_name, dpi=300, writer=PillowWriter(fps=time_multiplier*2))
    else:
        for i in frames:
            animate(i)
            plt.show()
    
        if save:
            for i in frames:
                idx = i
                if i < 0:
                    idx = len(reachable_set_snapshots) + i
                fig.savefig(figure_path + cl_system.dynamics.name + '_' + name + '_' + str(idx) + '.png', dpi=300, bbox_inches='tight')


def unified_plotter(cl_system, frames, experiments):        
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 20,
    })
    
    fig, ax = plt.subplots(figsize=(10,6))

    colors = {experiments[0]: orange, experiments[1]: blue, experiments[2]: '#AD00AB'} # 90008E 730071
    zorders = {experiments[0]: 1, experiments[1]: 3, experiments[2]: 2}
    legend_patches = []

    if cl_system.dynamics.name == "DoubleIntegrator":
        fs = 26
        plt.rcParams.update({"font.size": fs})
        plt.subplots_adjust(left=0.176, bottom=0.15, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        fig.set_size_inches(9.6, 7.2)
        constraint_color = '#262626'
        ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c=constraint_color, linewidth=2)
        rect = Rectangle(np.array([0.0, -1.25]), 3.25, 0.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
        ax.add_patch(rect)

        ax.plot(np.array([0, 0]), np.array([-1.25, 1.]), c=constraint_color, linewidth=2)
        rect = Rectangle(np.array([-1.5, -1.25]), 1.5, 2.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
        ax.add_patch(rect)

        ax.set_xlim([-0.5, 3.25])
        ax.set_ylim([-1.25, 0.5])

        ax.set_xlabel('$\mathbf{x}[0]$', fontsize=fs)
        ax.set_ylabel('$\mathbf{x}[1]$', fontsize=fs)

        linewidth = 1.5

        
    elif cl_system.dynamics.name == "Unicycle_NL":
        fig.set_size_inches(10, 5)
        delta = 0.
        # obstacles = [{'x': -10, 'y': -1, 'r': 3},
        #              {'x': -3, 'y': 2.5, 'r': 2 }]
        obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
                    {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
        for obstacle in obstacles:
            color = '#262626'
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor='none')
            ax.add_patch(circle)
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
            ax.add_patch(circle)

        ax.set_xlim([-10, 1])
        ax.set_ylim([-1.1, 4])
        ax.set_aspect('equal')

        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')

        

    for expr in experiments:

        data_file = dir_path + '/experimental_data/' + expr + '.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)

        num_steps = len(info[-1]["reachable_sets"]) - 1
        reachable_set_snapshots = []


        


        # info[-1]['time'] = 3
        for j, snapshot in enumerate(info):
            reachable_set_snapshot = []
            for i, reach_set_tuple in enumerate(snapshot['reachable_sets']):
                state_range = reach_set_tuple[0]
                is_symbolic = reach_set_tuple[1]
                collides = reach_set_tuple[2]
                # edgecolor = blue # '#2176FF'
    
                # if i == snapshot['child_idx'] and snapshot['child_idx'] != snapshot['parent_idx'] + 1:
                #     edgecolor = blue #'#D63230' #F45B69'
                
                # if i == 0:
                #     edgecolor = 'k'
                # elif is_symbolic:
                #     edgecolor = green # '#00CC00' # edgecolor = '#D112E2' # '#D14081' # '#53917E' # '#20A39E' # '#44BBA4'

                # if i == snapshot['parent_idx'] and j < len(info) - 1:
                #     edgecolor = magenta #'#FF00FF' # '#FFAE03'
                # if collides:
                #     edgecolor = orange # '#FF8000' # '#D63230'

                edgecolor = colors[expr]
                
                
                reachable_set_snapshot.append((state_range, edgecolor))
            
            # for i in range(num_times):
            reachable_set_snapshots.append(reachable_set_snapshot)
                


        # ax.clear()
        state_range = info[0]["reachable_sets"][0][0]
        num_trajectories = 500
        xs = sample_from_range(state_range, cl_system, num_steps, num_trajectories = num_trajectories)
        
        plot_exact_reachable_sets = False
        if plot_exact_reachable_sets:
            xs_sorted = xs.reshape((num_steps+1, num_trajectories, 3))
            for timestep_states in xs_sorted:
                xy = np.min(timestep_states, axis=0)
                width, height = (np.max(timestep_states, axis=0) - xy)[[0, 1]]
                rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
                ax.add_patch(rect)

        
        ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

        # xy = initial_set[[0, 1], 0]
        # width, height = initial_set[[0, 1], 1] - initial_set[[0, 1], 0]
        # rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect)
        
        

            
        
        linewidth = 0.9
        if expr == 'unicycle_CARV':
            linewidth = 1.0

        for reachable_set_snapshot in reachable_set_snapshots[-1]:
            set_range = reachable_set_snapshot[0]
            edgecolor = reachable_set_snapshot[1]
            xy = set_range[[0, 1], 0]
            width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
            rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor='none', zorder=zorders[expr])
            ax.add_patch(rect)
            alpha = 0.2
            if edgecolor == '#FF8000':
                alpha = 0.4
            if edgecolor == orange:
                alpha = 0.4

            if edgecolor != 'k':
                rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=edgecolor, alpha=alpha, zorder=zorders[expr])
                ax.add_patch(rect)
                if expr == 'unicycle_CARV':
                    label = 'CARV'
                elif expr == 'unicycle_hybr':
                    label = '\\texttt{unif}'
                else:
                    label = f'\\texttt{{{expr[9:]}}}'
                rect = Rectangle(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=edgecolor, alpha=1, zorder=zorders[expr], label=label)
        
        legend_patches.append(rect)

    legend_patches[1], legend_patches[0] = legend_patches[0], legend_patches[1]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=16, framealpha=1)
    # for i in frames:
        # animate(i)

    plt.show()


def sample_from_range(state_range, cl_system, num_steps, num_trajectories = 100):
    np.random.seed(0)
    num_states = cl_system.At.shape[0]
    

    x0s = np.random.uniform(
        low=state_range[:, 0],
        high=state_range[:, 1],
        size=(num_trajectories, num_states),
    )
    

    xs = x0s
    
    xt = xs
    for _ in range(num_steps):
        u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller.cpu())
        xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
        xt = xt1

        xs = np.vstack((xs, xt1))
    
    return xs

def plot_comparison(info):
    smarkersize = 40
    omarkersize = 70

    carv_k_max = info["carv_k_max"]
    carv_time = info["carv_time"]
    carv_error = info["carv_error"]
    carv_verif_results = info["carv_verif_results"]
    
    ttt_k_max = info["ttt_k_max"]
    ttt_time = info["ttt_time"]
    ttt_error = info["ttt_error"]
    ttt_verif_results = info["ttt_verif_results"]
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 20,
    })

    
    for y_axis in ["time", "error"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
        plt.subplots_adjust(left=0.137, bottom=0.194, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

        if y_axis == "time":
            carv_y = carv_time
            ttt_y = ttt_time
            ax.set_ylim([5, 34])
        elif y_axis == "error":
            carv_y = carv_error
            ttt_y = ttt_error
            ax.set_yscale('log')

        
        carv_label, = ax.plot(carv_k_max, carv_y, c=lightblue, zorder=3) # 2176FF
        ttt_label, = ax.plot(ttt_k_max, ttt_y, c=lightorange, zorder=1) # FF8000 FF00FF 00CC00
        blue_o_marker = ax.scatter(carv_k_max[-1], carv_y[-1], marker='o', c=blue, s=omarkersize, zorder=0)
        orange_o_marker = ax.scatter(carv_k_max[-1], carv_y[-1], marker='o', c=orange, s=omarkersize, zorder=0)
        blue_x_marker = ax.scatter(ttt_k_max[-1], ttt_y[-1], marker='s', c=blue, s=smarkersize, zorder=0)
        orange_x_marker = ax.scatter(ttt_k_max[-1], ttt_y[-1], marker='s', c=orange, s=smarkersize, zorder=0)

        for idx, k in enumerate(carv_k_max):
            marker = 'o'
            color = blue# '#1857BD' # '#14489C'
            markersize = omarkersize
            if not carv_verif_results[idx]:
                marker = 's'
                color = blue
                markersize = smarkersize
            ax.scatter(k, carv_y[idx], marker=marker, c=color, s=markersize, zorder=4)
        
        for idx, k in enumerate(ttt_k_max):
            marker = 'o'
            color = orange # '#C96500'
            markersize = omarkersize
            if not ttt_verif_results[idx]:
                marker = 's'
                color = orange # CC00CC 008F00
                markersize = smarkersize
            ax.scatter(k, ttt_y[idx], marker=marker, c=color, s=markersize, zorder=2)
        
        ax.set_xlabel('$k_{max}$')
        if y_axis == "time":
            ax.set_ylabel('Computation Time [s]')
        elif y_axis == "error":
            ax.set_ylabel('Approximation Error')

        ax.legend([carv_label, ttt_label, (blue_o_marker, orange_o_marker), (blue_x_marker, orange_x_marker)], ['CARV', '$\\mathtt{unif}$', 'verified safe', 'verification failed'], fontsize=16, handler_map={tuple: HandlerTuple(ndivide=None)})
        ax.set_xticks([6, 12, 18, 24])
        plt.show()


def plot_sweep_constraint(info, system):

    if system == "double_integrator":
        carv30_time = info["carv14_time"]
        carv30_delta = info["carv14_delta"]
        carv30_verif_results = info["carv14_verif_results"]
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            "font.size": 20,
        })
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
        plt.subplots_adjust(left=0.137, bottom=0.194, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        # import pdb; pdb.set_trace()
        carv30_line, = ax.plot(carv30_delta, carv30_time, c='#2176FF')
        symb_line, = ax.plot([0, 0.19], [25.82, 25.82], c='#00CC00')
        part_delta = np.hstack((np.arange(0, 0.12, 0.01))) #0.15))
        part_time = [18.81, #0
                    25.57, 
                    25.57, 
                    25.57, 
                    30.87, 
                    30.87, #5
                    36.23,
                    39.32,
                    39.32,
                    47.15,
                    50.63, #10
                    67.88, # 17 didn't work; trying 18; 18 worked
                    #251.44 - 0.15
                    ]
        part_line, = ax.plot(part_delta, part_time, c='#FF00FF')
        # ax.plot([0.11, 0.11], [0, 150], 'k--')
        # ax.plot([0.002, 0.002], [0, 150], 'k--')
        # ax.plot([0.38, 0.38], [0, 275], 'k--')


        ax.set_ylabel('Computation Time [s]')
        ax.set_xlabel('$\\Delta c$ [m]')
        # ax.set_xlim([-0.01, 0.28])
        # ax.set_ylim([-5, 280])

        ax.legend([part_line, symb_line, carv30_line], ['$\mathtt{part}$', '$\mathtt{symb}$', 'CARV'], loc='upper right', framealpha=1, fontsize=16)

        plt.show()

    elif system == "unicycle":
        carv10_time = info["carv10_time"]
        carv10_delta = info["carv10_delta"]
        carv10_verif_results = info["carv10_verif_results"]
        
        carv14_time = info["carv14_time"]
        carv14_delta = info["carv14_delta"]
        carv14_verif_results = info["carv14_verif_results"]

        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            "font.size": 20,
        })

        # prev color scheme:
        # blue: #1857BD
        # green: #008F00
        # magenta: #C200C2
        # orange: #C96500

        # blue = '#0072B2'
        # green = '#009E73'
        # orange = '#D55E00'
        # magenta = '#CC79A7'


        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
        plt.subplots_adjust(left=0.137, bottom=0.194, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
        # import pdb; pdb.set_trace()

        # Submission color scheme:
        # blue: #0072B2
        # green: #009E73
        # magenta: #CC79A7
        # orange: #D55E00

        # Revised color scheme:
        blue = '#2E72F2' # '#5087F5' # '#005AB5'# '#1C44FE'
        green = '#28DD51'
        orange = '#FB762F'
        purple = '#AD00AB'


        carv10_line, = ax.semilogy(carv10_delta, carv10_time, c=blue, linewidth=2)
        carv14_line, = ax.semilogy(carv14_delta, carv14_time, c=green, linewidth=2)
        part_line, = ax.plot([0.0, 0.11], [525, 525], c=purple, linewidth=2)
        hybr_line, = ax.plot([0.0, 0.002], [7.20, 7.20], c=orange, linewidth=2)
        
        alpha = 0.8
        ax.plot([carv10_delta[-1], carv10_delta[-1]], [5, 900], c=blue, linestyle='--', alpha=alpha)
        ax.plot([carv14_delta[-1], carv14_delta[-1]], [5, 900], c=green, linestyle='--', alpha=alpha)
        ax.plot([0.11, 0.11], [5, 900], c=purple, linestyle='--', alpha=alpha)
        ax.plot([0.002, 0.002], [5, 900], c=orange, linestyle='--', alpha=alpha)
        # ax.plot([0.33, 0.33], [5, 900], c='k', linestyle='--')
        
        # ax.plot([0.38, 0.38], [0, 275], 'k--')


        ax.set_ylabel('Computation Time [s]')
        ax.set_xlabel('$\\Delta r$ [m]')
        ax.set_xlim([-0.01, 0.32])
        # ax.set_ylim([-5, 375])
        # plt.yscale('log')

        # plt.annotate('best $\\mathtt{hybr}$', (0.007, 130), fontsize=20)
        # plt.annotate('best $\\mathtt{part}$', (0.115, 130), fontsize=20)

        ax.legend([part_line, carv14_line, carv10_line, hybr_line], 
                  ['$\\mathtt{part}$', '$\\mathrm{CARV}_{24}$', '$\\mathrm{CARV}_{10}$', '$\\mathtt{unif}$'],
                  loc='upper right',
                  framealpha=1,
                  fontsize=16,
                #   bbox_to_anchor=(1.12, 0.0)
                )
        plt.show()

if __name__ == "__main__":

    # experiment = "double_integrator"
    # experiment = "unicycle"
    # experiment = "quadrotor_gates"
    experiment = "unified_unicycle"
    # experiment = "comparison"
    # experiment = "sweep_unicycle_constraints"
    # experiment = "sweep_double_integrator_constraints"

    if experiment == "double_integrator":
        data_file = dir_path + '/experimental_data/double_integrator_CARV.pkl'

        with open(data_file, 'rb') as f:
            info = pickle.load(f)
        
        controller = load_controller("DoubleIntegrator", "constraint_default_more_data_5hz", False, device='cpu')
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        cl_system = cl_systems.ClosedLoopDynamics(controller, ol_dyn)
        frames = [4, 9, -1]
        for i in frames:
            official_plotter(info, cl_system, [i])

    elif experiment == "unicycle":
        data_file = dir_path + '/experimental_data/unicycle_hybr.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)
        
        controller = load_controller("Unicycle_NL", "natural_none_default", False, device='cpu')
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        cl_system = cl_systems.Unicycle_NL(controller, ol_dyn)
        frames = [-1]
        for i in frames:
                official_plotter(info, cl_system, [i])

    elif experiment == "quadrotor_gates":
        data_file = dir_path + '/experimental_data/quadrotor_gates.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)
        
        controller = load_controller("Quadrotor_NL", "natural_none_gates", False, device='cpu')
        ol_dyn = dynamics.Quadrotor_NL(dt=0.2)
        cl_system = cl_systems.Quadrotor(controller, ol_dyn)
        frames = []
        for i in frames:
                official_3D_plotter(info, cl_system, [i], save=True, name='CARV15')
        
        official_3D_plotter(info, cl_system, save_animation=True, name='CARV15_pan_adjusted')
    
    elif experiment == "unified_unicycle":
        experiments = ["unicycle_hybr", "unicycle_CARV", "unicycle_part"]
        controller = load_controller("Unicycle_NL", "natural_none_default", False, device='cpu')
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        cl_system = cl_systems.Unicycle_NL(controller, ol_dyn)
        frames = [-1]
        for i in frames:
                unified_plotter(cl_system, [i], experiments)

    elif experiment == "comparison":
        # data_file = dir_path + '/experimental_data/comparison_short.pkl'
        data_file = dir_path + '/experimental_data/sweep_k_amended.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)
        
        plot_comparison(info)

    elif experiment == "sweep_unicycle_constraints":
        data_file = dir_path + '/experimental_data/sweep_unicycle_constraint.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)

        data_file = dir_path + '/experimental_data/sweep_unicycle_constraint_0.29.pkl'
        with open(data_file, 'rb') as f:
            info2 = pickle.load(f)

        # info["carv14_time"] = info2["carv14_time"]
        # info["carv14_delta"] = info2["carv14_delta"]
        # info["carv14_verif_results"] = info2["carv14_verif_results"]

        plot_sweep_constraint(info2, system="unicycle")

    elif experiment == "sweep_double_integrator_constraints":
        data_file = dir_path + '/experimental_data/sweep_double_integrator_constraint.pkl'
        with open(data_file, 'rb') as f:
            info = pickle.load(f)
        
    
        
        plot_sweep_constraint(info, system="double_integrator")


    