"""Implementation of TTT algorithm"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from ast import literal_eval
from itertools import product
from copy import deepcopy
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum
import nfl_veripy.dynamics as dynamics

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import cl_systems

from utils.nn import load_controller
from utils.robust_training_utils import ReachableSet
from utils.robust_training_utils import Analyzer

from real_reachset_sim import CalculationType
from real_reachset_sim import ReachabilityTester
from real_reachset_sim import ReachableSetHorizon


