from cl_systems.double_integrator import *
from cl_systems.unicycle import *
from cl_systems.cl_dynamics import *


Controllers = {
    'di_2layer': di_2layer_controller,
    'di_3layer': di_3layer_controller,
    'di_4layer': di_4layer_controller,
    'unicycle_nl_4layer': unicycle_4layer_controller,
}

ClosedLoopDynamics = ClosedLoopDynamics
