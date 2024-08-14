"""
A simple script to train certified defense using the auto_LiRPA library.

We compute output bounds under input perturbations using auto_LiRPA, and use
them to form a "robust loss" for certified defense.  Several different bound
options are supported, such as IBP, CROWN, and CROWN-IBP. This is a basic
example on MNIST and CIFAR-10 datasets with Lp (p>=0) norm perturbation. For
faster training, please see our examples with loss fusion such as
cifar_training.py and tinyimagenet_training.py
"""

import pdb
import nfl_veripy.dynamics as dynamics

import time
import torch
import random
import multiprocessing
import argparse
import os
import torch.optim as optim
from torch.nn import MSELoss
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
# import models
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

from nfl_veripy.utils.nn_closed_loop import *
import cl_systems
from _static.dataloaders.double_integrator_loader import double_integrator_loaders, DIDataset
from _static.dataloaders.unicycle_nl_loader import unicycle_nl_loaders, UniDataset
from utils.robust_training_utils import calculate_reachable_sets
from utils.robust_training_utils import Analyzer

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="default", choices=["MNIST", "CIFAR", "default", "expanded", "expanded_5hz", "dagger", "default_5hz", "default_more_data_5hz"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.3, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="resnet", help='model name (mlp_3layer, cnn_4layer, cnn_6layer, cnn_7layer, resnet)')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=3,length=60", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument("--conv_mode", type=str, choices=["matrix", "patches"], default="patches")
parser.add_argument("--save_model", type=str, default='')
parser.add_argument("--system", type=str, choices=["double_integrator", "Unicycle_NL"], default='double_integrator')
parser.add_argument("--training_method", type=str, choices=["natural", "robust", "constraint", "robust-constraint"], default='natural')
parser.add_argument("--refinement_method", type=str, choices=["none", "smart-partition", "uniform-partition", "symbolic_indices", "smart-partition-recalculate"], default='none')
# parser.add_argument("--constraint", type=str, choices=["none", "data", "bounds", "all"], default='none')

args = parser.parse_args()

def constraint_loss(output, boundary):
    violation = torch.nn.ReLU()(-output + boundary)
    # if any(output < -1):
    #     print(violation)
    # import pdb; pdb.set_trace()

    # loss = MSELoss()(violation, torch.zeros(output.shape[0]))
    loss = torch.sum(violation)

    return loss

def condition(input_range):
    return input_range[1, 0] >= -1

def Train_Regressor(model, cl_system, t, loader, eps_scheduler, norm, train, opt, bound_type, method='natural', device='cpu', constriant='none', analyzer=None):
    num_class = 1
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-6:
            eps = 0.2
            # batch_method = "natural"
        if t < 80:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        
        
        # generate specifications
        c = torch.eye(num_class).type_as(data).unsqueeze(0)
        if list(model.parameters())[0].is_cuda: # TODO: figure out how to run with cuda
            c = c.cuda()
        # remove specifications to self
        # I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        # c = (c[I].view(data.size(0), num_class - 1, num_class))
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        # if norm > 0:
        #     ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        # elif norm == 0:
        #     ptb = PerturbationL0Norm(eps = eps_scheduler.get_max_eps(), ratio = eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
        ptb = PerturbationLpNorm(eps = eps)
        x = BoundedTensor(data, ptb)
        output = model(x)
        regular_loss = MSELoss()(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('MSE', regular_loss.item(), x.size(0))
        # import pdb; pdb.set_trace()   
        # meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

        if batch_method == "robust" or batch_method == "robust-constraint" or batch_method == "constraint":
            if bound_type == "IBP":
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
            elif bound_type == "CROWN":
                # init_ranges = np.array([
                #     [
                #         [2.5, 2.75],
                #         [-0.25, 0]
                #     ],
                #     [
                #         [2.75, 3.0],
                #         [-0.25, 0]
                #     ],
                #     [
                #         [2.5, 3.0],
                #         [0, 0.25]
                #     ],
                #     # [
                #     #     [2.5, 3.],
                #     #     [-0.25, 0.25]
                #     # ]
                # ])
                # reach_sets = torch.zeros((len(init_ranges), 25, 2, 2))
                # for i, init_range in enumerate(init_ranges):
                #     reach_sets[i] = calculate_reachable_sets(cl_system, init_range, 25)
                #     # lb, ub = cl_system.compute_bounds(x=(x,), method="backward")
                #     # lb, ub = cl_system.compute_bounds(x=(x,), method=None, IBP=True)
                # lb = reach_sets[:, :, :, 0]
                # ub = reach_sets[:, :, :, 1]
                
                analyzer.calculate_reachable_sets()
                # analyzer.calculate_N_step_reachable_sets(indices=[3, 4, 5, 6, 7])
                # import pdb; pdb.set_trace()
                all_sets = analyzer.get_all_ranges()
                lb, ub = all_sets[:, :, 0], all_sets[:, :, 1]
                analyzer.switch_sets_on_off(condition)
                if t%10 == 0:
                    analyzer.switch_sets_on_off(lambda x: False)
            elif bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = model.compute_bounds(x=(x,), IBP=True, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                    lb = clb * factor + ilb * (1 - factor)

                
            elif bound_type == "CROWN-FAST":
                # Similar to CROWN-IBP but no mix between IBP and CROWN bounds.
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)


            robust_loss = MSELoss()(lb, ub)

            # Pad zero at the beginning for each example, and use fake label "0" for all examples
            # lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            # fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            # robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
        
        alpha = 1
        beta = 0.01
        if batch_method == "robust":
            loss = regular_loss + alpha*robust_loss
        elif batch_method == "constraint":
            violation_loss = constraint_loss(lb[:, 1], -1)
            loss = regular_loss + beta*violation_loss
        elif batch_method == "robust-constraint":
            violation_loss = constraint_loss(lb[:, 1], -1)
            loss = regular_loss + alpha*robust_loss + beta*violation_loss
        elif batch_method == "natural":
            loss = regular_loss
        
        if train:
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_loss.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))
        if batch_method != "natural":
            meter.update('Robust_CE', robust_loss.item(), data.size(0))
            # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            # If any margin is < 0 this example is counted as an error
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

# def Train_Regressor_crown(model, t, loader, eps_scheduler, norm, train, opt, method):
#     # if train=True, use training mode
#     # if train=False, use test mode, no back prop
    
#     num_class = 1
#     meter = MultiAverageMeter()
#     batch_multiplier = 1

#     if train:
#         model.train()
#         eps_scheduler.train()
#         eps_scheduler.step_epoch()
#         eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
#     else:
#         model.eval()
#         eps_scheduler.eval()
#     kappa = 1
#     beta = 1

#     if train:
#         model.train() 
#     else:
#         model.eval()
#     # pregenerate the array for specifications, will be used for scatter
#     sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
#     for i in range(sa.shape[0]):
#         for j in range(sa.shape[1]):
#             if j < i:
#                 sa[i][j] = j
#             else:
#                 sa[i][j] = j + 1
#     sa = torch.LongTensor(sa) 
#     batch_size = loader.batch_size * batch_multiplier
#     if batch_multiplier > 1 and train:
#         print('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
#     # per-channel std and mean
#     std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#     mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
 
#     model_range = 0.0
#     end_eps = eps_scheduler.get_eps(t+1, 0)
#     if end_eps < np.finfo(np.float32).tiny:
#         print('eps {} close to 0, using natural training'.format(end_eps))
#         method = "natural"
#     for i, (data, labels) in enumerate(loader): 
#         start = time.time()
#         eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
#         if train and i % batch_multiplier == 0:   
#             opt.zero_grad()
#         # generate specifications

#         c = torch.eye(num_class).type_as(data).unsqueeze(0).cuda()

#         data = data.cuda()

#         output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural"))

#         # import pdb; pdb.set_trace()
#         loss = 0.01*MSELoss()(output, labels.cuda())

#         data_ub = data + (eps / std.cuda())
#         data_lb = data - (eps / std.cuda())

#         # data_ub, data_lb = data_ub.cuda(), data_lb.cuda()

#         # import pdb; pdb.set_trace()
#         ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub[0, 0], x_L=data_lb[0, 0], eps=eps, C=c, method_opt="interval_range")


#         lamb = 1
#         loss += lamb*MSELoss()(ub, ilb)
#         if train:
#             loss.backward()
#             if i % batch_multiplier == 0 or i == len(loader) - 1:
#                 opt.step()

def collect_dagger_dataset(pi_i, dyn, x0s, beta, num_steps):
    xt = x0s
    xs = x0s
    us = None
    for _ in range(num_steps):
        u_exp = dyn.control_mpc(xt)
        u_nn = dyn.control_nn(xt, pi_i)
        ut = beta*u_exp + (1 - beta)*u_nn
        xt1 = dyn.dynamics_step(xt, ut)
        xt = xt1

        xs = np.vstack((xs, xt1))
        if us is None:
            us = u_exp
        else:
            us = np.vstack((us, u_exp))
    
    u_exp = dyn.control_mpc(xt)
    us = np.vstack((us, u_exp))

    return xs, us


def Train_Dagger(test_loader, args):
    num_trajectories = 50
    beta = 1
    p = 0.5
    num_iters = 10
    init_range = np.array([
        [2.5, 3.0],
        [-0.25, 0.25]
    ])
    ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
    controller_name = "di_4layer"
    
    neurons_per_layer = [15, 10, 5]
    pi_0 = cl_systems.Controllers["di_4layer"](neurons_per_layer)
    dummy_input = torch.tensor([[2.75, 0.]], device=args.device)
    policies = [BoundedModule(pi_0, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)]
    xs_all = None
    us_all = None

    for i in range(num_iters):
        
        x0s = np.random.uniform(
                    low=init_range[:, 0],
                    high=init_range[:, 1],
                    size=(num_trajectories, ol_dyn.At.shape[0]),
                )
        xs, us = collect_dagger_dataset(policies[-1], ol_dyn, x0s, beta, num_steps=25)

        if xs_all is None:
            xs_all = xs
            us_all = us
        else:
            xs_all = np.vstack((xs_all, xs))
            us_all = np.vstack((us_all, us))

        #################################################
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12, 12))
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "Helvetica"
        # })
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xs_all[:, 0], xs_all[:, 1], us_all.flatten(), c='b')
        # ax.scatter(xs[:, 0], xs[:, 1], us.flatten(), c='r')
        # ax.set_xlabel('x1', fontsize=20)
        # ax.set_ylabel('x2', fontsize=20)
        # ax.set_zlabel('u', fontsize=20)
        # plt.show()
        #################################################
        
        pi_i = cl_systems.Controllers["di_4layer"](neurons_per_layer).to(args.device)
        model = BoundedModule(pi_i, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
        cl_dyn = cl_systems.ClosedLoopDynamics(pi_i, ol_dyn).to(args.device)
        bounded_cl_sys = BoundedModule(cl_dyn, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)

        dataset = DIDataset(torch.tensor(xs_all, dtype=torch.float32), torch.tensor(us_all, dtype=torch.float32), transform=None)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        loader.mean = torch.tensor([0.])
        loader.std = torch.tensor([1.])

        norm = float(args.norm)
        opt = optim.Adam(pi_i.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)

        # train_loader, val_loader, test_loader = double_integrator_loaders(batch_size=64, dataset_name=args.data)
        # train_loader.mean = val_loader.mean = test_loader.mean = torch.tensor([0.])
        # train_loader.std = val_loader.std = test_loader.std = torch.tensor([1.])

        timer = 0.0
        print("############################################ TRAINING ITERATION {} ############################################".format(i))
        for t in range(1, args.num_epochs+1):
            if eps_scheduler.reached_max_eps():
                # Only decay learning rate after reaching the maximum eps
                lr_scheduler.step()
            print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
            start_time = time.time()
            Train_Regressor(model, bounded_cl_sys, t, loader, eps_scheduler, norm, True, opt, args.bound_type, method=args.training_method)
            epoch_time = time.time() - start_time
            timer += epoch_time
            print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            print("Evaluating...")
            with torch.no_grad():
                Train_Regressor(model, bounded_cl_sys, t, test_loader, eps_scheduler, norm, False, None, args.bound_type)

            # import pdb; pdb.set_trace()
            path = os.getcwd() + '/nfl_robustness_training/src/controller_models/'
            model_file = path + args.system + '/daggers/' + controller_name + '/' + args.training_method + '_' + args.data + '_' + '{}'.format(i) + '.pth'
            torch.save({'state_dict': model.state_dict(), 'epoch': t}, model_file)

        beta *= p
        policies.append(model)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    if args.system == 'double_integrator':
        controller_name = "di_4layer"
        neurons_per_layer = [30, 20, 10]
        controller_ori = cl_systems.Controllers[controller_name](neurons_per_layer)
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(args.device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(args.device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(args.device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller_ori, ol_dyn).to(args.device)

        ## Step 2: Prepare dataset as usual
        if args.data in ['default', 'expanded', 'expanded_5hz', 'default_5hz', 'default_more_data_5hz']:
            train_loader, val_loader, test_loader = double_integrator_loaders(batch_size=64, dataset_name=args.data)
            train_loader.mean = val_loader.mean = test_loader.mean = torch.tensor([0.])
            train_loader.std = val_loader.std = test_loader.std = torch.tensor([1.])
            dummy_input = torch.tensor([[2.75, 0.]], device=args.device)
            cl_dyn(dummy_input)

        
            ## Step 3: wrap model with auto_LiRPA
            # The second parameter dummy_input is for constructing the trace of the computational graph.
            model = BoundedModule(controller_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
            bounded_cl_sys = BoundedModule(cl_dyn, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
            num_steps = 25
            init_ranges = torch.tensor([
                [2.5, 3.],
                [-0.25, 0.25]
            ], device=args.device)
            analyzer = Analyzer(cl_dyn, num_steps, init_ranges, device=args.device)
            if args.refinement_method in ["smart-partition", "smart-partition-recalculate"]:
                analyzer.set_partition_strategy(0, np.array([3,3]))
                # analyzer.set_partition_strategy(7, np.array([3,3]))
                # analyzer.set_partition_strategy(12, np.array([2,2]))
        elif args.data == "dagger":
            train_loader, val_loader, test_loader = double_integrator_loaders(batch_size=64, dataset_name="expanded")
            train_loader.mean = val_loader.mean = test_loader.mean = torch.tensor([0.])
            train_loader.std = val_loader.std = test_loader.std = torch.tensor([1.])
            Train_Dagger(test_loader, args)
            
            # analyzer.calculate_N_step_reachable_sets(indices=None)
            # analyzer.plot_reachable_sets()
            # init_range = np.array([
            #     [2.5, 3.0],
            #     [-0.25, 0.25]
            # ])
            # from utils.robust_training_utils import calculate_reachable_set
            # calculate_reachable_set(bounded_cl_sys, init_range, 25)
            
            # eps = torch.tensor([0.3, 0.4])
            # ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
            # bounded_input = BoundedTensor(dummy_input, ptb)
            # import pdb; pdb.set_trace()
            # lb, ub = bounded_cl_sys.compute_bounds(x = (bounded_input,), method = "backward")
            # import pdb; pdb.set_trace()
            # dummy_input_2 = torch.tensor([[2.75, 0.], [3., 0.], [2.5, 0.], [2.75, 0.1]])
            # ptb_2 = PerturbationLpNorm(norm=np.inf, eps=0.)
            # bounded_input_2 = BoundedTensor(dummy_input_2, ptb_2)
            # lb, ub = bounded_cl_sys.compute_bounds(x=(bounded_input_2,), method="backward")
            # import pdb; pdb.set_trace()


    if args.system == 'Unicycle_NL':
        controller_name = "unicycle_nl_4layer"
        neurons_per_layer = [40, 20, 10]
        mean = torch.tensor([-7.5, 2.5, 0], device=args.device)
        std = torch.tensor([7.5, 2.5, torch.pi/6], device=args.device)
        controller_ori = cl_systems.Controllers[controller_name](neurons_per_layer, mean, std)
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(args.device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(args.device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(args.device)
        cl_dyn = cl_systems.Unicycle_NL(controller_ori, ol_dyn).to(args.device)

        ## Step 2: Prepare dataset as usual
        if args.data in ['default', 'expanded', 'expanded_5hz', 'default_5hz', 'default_more_data_5hz']:
            train_loader, val_loader, test_loader = unicycle_nl_loaders(batch_size=64, dataset_name=args.data)
            train_loader.mean = val_loader.mean = test_loader.mean = torch.tensor([0.])
            train_loader.std = val_loader.std = test_loader.std = torch.tensor([1.])
            dummy_input = torch.tensor([[-12.5, 3.5, 0.]], device=args.device)
            cl_dyn(dummy_input)

        
            ## Step 3: wrap model with auto_LiRPA
            # The second parameter dummy_input is for constructing the trace of the computational graph.
            model = BoundedModule(controller_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
            bounded_cl_sys = BoundedModule(cl_dyn, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
            num_steps = 25
            init_ranges = torch.tensor([
                [-15., -14.],
                [4., 5.],
                [-np.pi/6, np.pi/6]
            ], device=args.device)
            analyzer = Analyzer(cl_dyn, num_steps, init_ranges, device=args.device)
            if args.refinement_method in ["smart-partition", "smart-partition-recalculate"]:
                analyzer.set_partition_strategy(0, np.array([3,3]))
    

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(controller_ori.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    print("Model structure: \n", str(cl_dyn))

    timer = 0.0
    for t in range(1, args.num_epochs+1):
        if eps_scheduler.reached_max_eps():
            # Only decay learning rate after reaching the maximum eps
            lr_scheduler.step()
        print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
        start_time = time.time()
        Train_Regressor(model, bounded_cl_sys, t, train_loader, eps_scheduler, norm, True, opt, args.bound_type, method=args.training_method, analyzer=analyzer)
        epoch_time = time.time() - start_time
        timer += epoch_time
        print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
        print("Evaluating...")
        with torch.no_grad():
            Train_Regressor(model, bounded_cl_sys, t, test_loader, eps_scheduler, norm, False, None, args.bound_type, analyzer=analyzer)

        # import pdb; pdb.set_trace()
        path = os.getcwd() + '/nfl_robustness_training/src/controller_models/'
        model_file = path + args.system + '/' + controller_name + '/' + args.training_method + '_' + args.refinement_method + '_' + args.data + '.pth'
        torch.save({'state_dict': cl_dyn.controller.state_dict(), 'epoch': t}, model_file)

        # bounded_cl_sys = BoundedModule(cl_dyn, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
        
        # init_range = np.array([
        #     [2.5, 3.0],
        #     [-0.25, 0.25]
        # ])
        # from utils.robust_training_utils import calculate_reachable_sets
        # calculate_reachable_set(bounded_cl_sys, init_range, 25)
        


    


    # ## Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    # if args.data == 'MNIST':
    #     model_ori = models.Models[args.model](in_ch=1, in_dim=28)
    # else:
    #     model_ori = models.Models[args.model](in_ch=3, in_dim=32)
    # if args.load:
    #     state_dict = torch.load(args.load)['state_dict']
    #     model_ori.load_state_dict(state_dict)

    # ## Step 2: Prepare dataset as usual
    # if args.data == 'MNIST':
    #     dummy_input = torch.randn(2, 1, 28, 28)
    #     train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    #     test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    # elif args.data == 'CIFAR':
    #     dummy_input = torch.randn(2, 3, 32, 32)
    #     normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
    #     train_data = datasets.CIFAR10("./data", train=True, download=True,
    #             transform=transforms.Compose([
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.RandomCrop(32, 4),
    #                 transforms.ToTensor(),
    #                 normalize]))
    #     test_data = datasets.CIFAR10("./data", train=False, download=True, 
    #             transform=transforms.Compose([transforms.ToTensor(), normalize]))

    # train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    # test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    # if args.data == 'MNIST':
    #     train_data.mean = test_data.mean = torch.tensor([0.0])
    #     train_data.std = test_data.std = torch.tensor([1.0])
    # elif args.data == 'CIFAR':
    #     train_data.mean = test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
    #     train_data.std = test_data.std = torch.tensor([0.2023, 0.1994, 0.2010])

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    # model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)

    # ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    # opt = optim.Adam(model.parameters(), lr=args.lr)
    # norm = float(args.norm)
    # lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    # eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    # print("Model structure: \n", str(model_ori))

    # ## Step 5: start training
    # if args.verify:
    #     eps_scheduler = FixedScheduler(args.eps)
    #     with torch.no_grad():
    #         Train(model, 1, test_data, eps_scheduler, norm, False, None, args.bound_type)
    # else:
    #     timer = 0.0
    #     for t in range(1, args.num_epochs+1):
    #         if eps_scheduler.reached_max_eps():
    #             # Only decay learning rate after reaching the maximum eps
    #             lr_scheduler.step()
    #         print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
    #         start_time = time.time()
    #         Train(model, t, train_data, eps_scheduler, norm, True, opt, args.bound_type)
    #         epoch_time = time.time() - start_time
    #         timer += epoch_time
    #         print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
    #         print("Evaluating...")
    #         with torch.no_grad():
    #             Train(model, t, test_data, eps_scheduler, norm, False, None, args.bound_type)
    #         torch.save({'state_dict': model_ori.state_dict(), 'epoch': t}, args.save_model if args.save_model != "" else args.model)


if __name__ == "__main__":
    main(args)