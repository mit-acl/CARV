import torch
import os


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from nn import load_controller
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()


if len(sys.argv) < 3:
    print("Usage: python pth2pt.py <system_name> <controller_name>")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Save controller state dict to a file.")
    parser.add_argument("system_name", type=str, help="Name of the system")
    parser.add_argument("controller_name", type=str, help="Name of the controller")
    args = parser.parse_args()

    device = 'cpu'
    controller = load_controller(args.system_name, args.controller_name, False, device=device)
    save_controller(controller, f"{args.system_name}_{args.controller_name}.pt")

def save_controller(controller, filename="controller.pt"):
    save_path = os.path.join(cwd, filename)
    torch.save(controller.state_dict(), save_path)
    print(f"Controller saved to {save_path}")


if __name__ == "__main__":
    main()