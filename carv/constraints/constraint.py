import torch

def di_condition(input_range):
        delta = 0.0
        return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + delta
    
def unicycle_condition(input_range):
    delta = 0.29
    obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
                    {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
    
    rx, ry = input_range[[0, 1], 0]
    width, height = input_range[[0, 1], 1] - input_range[[0, 1], 0]

    for obs in obstacles:
        cx, cy = obs['x'], obs['y']
        testX = torch.tensor(cx)
        testY = torch.tensor(cy)

        if (cx < rx):
            testX = rx
        elif (cx > rx + width):
            testX = rx + width


        if (cy < ry):
            testY = ry
        elif (cy > ry + height):
            testY = ry + height
        
        dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
        if dist < obs['r']:
            return False
        
    return True

def quadrotor_condition(input_range):
    delta = 0.0
    # obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
    #              {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
    # yoffset1 = 2
    # zoffset1 = 2
    # yoffset2 = -1.5
    # zoffset2 = 0.25
    # little_radius = 1.25 * 0.5
    # big_radius = 2.5
    # yoffset1 = 1
    # zoffset1 = 3
    # yoffset2 = -1.5
    # zoffset2 = 1
    # little_radius = 1.25*0.75
    # big_radius = 0.3
    # yoffset3 = 0
    # zoffset3 = 0
    # obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
    #              {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
    #             #  {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
    #             #  {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
    #              {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
    #              {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
    #             #  {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
    #             #  {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
    
    #              {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
    #              {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
    #              {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
    #              {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
    #              {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
    #              {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
    #              {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
    #              {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},
                    
    #              {'x': 0, 'y': -2. + yoffset3, 'r': little_radius},
    #              {'x': 0, 'y': 2. + yoffset3, 'r': little_radius},
    #              {'x': 0, 'z': -1. + zoffset3, 'r': little_radius},
    #              {'x': 0, 'z': 3. + zoffset3, 'r': little_radius},]
    
    yoffset1 = 1
    zoffset1 = 3
    yoffset2 = -1.5
    zoffset2 = 1
    little_radius = 1.25*0.4
    big_radius = 0.03
    yoffset3 = 0
    zoffset3 = 0
    obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
                    {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
                #  {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
                #  {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
                    {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
                    {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
                #  {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
                #  {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
    
                    {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
                    {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
                    {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
                    {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
                    {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
                    {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
                    {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
                    {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},
                    
                    {'x': 0, 'y': -2. + yoffset3, 'r': little_radius},
                    {'x': 0, 'y': 2. + yoffset3, 'r': little_radius},
                    {'x': 0, 'z': -1. + zoffset3, 'r': little_radius},
                    {'x': 0, 'z': 3. + zoffset3, 'r': little_radius},]
    
    rx, ry, rz = input_range[[0, 1, 2], 0]
    length, width, height = input_range[[0, 1, 2], 1] - input_range[[0, 1, 2], 0]

    for obs in obstacles:
        if 'y' in obs:
            cx, cy = obs['x'], obs['y']
            testX = torch.tensor(cx)
            testY = torch.tensor(cy)

            if (cx < rx):
                testX = rx
            elif (cx > rx + length):
                testX = rx + length


            if (cy < ry):
                testY = ry
            elif (cy > ry + width):
                testY = ry + width
            
            dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
            if dist < obs['r']:
                return False
        if 'z' in obs:
            cx, cz = obs['x'], obs['z']
            testX = torch.tensor(cx)
            testZ = torch.tensor(cz)

            if (cx < rx):
                testX = rx
            elif (cx > rx + length):
                testX = rx + length


            if (cz < rz):
                testZ = rz
            elif (cz > rz + height):
                testZ = rz + height
            
            dist = torch.sqrt((cx-testX)**2 + (cz - testZ)**2)
            if dist < obs['r']:
                return False
        
    return True

class Constraint:
    def __init__(self, perturbation=0.0):
        self.perturbation = perturbation

    def is_safe(self, input_range):
        raise NotImplementedError("Subclasses should implement this method")
    
    def plot(self, input_range):
        raise NotImplementedError("Subclasses should implement this method")

# class DIConstraint(Constraint):
#     def is_safe(self, input_range):
#         delta = self.perturbation
#         return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + delta

# class UnicycleConstraint(Constraint):
#     def is_safe(self, input_range):
#         delta = self.perturbation
#         obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
#                         {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
        
#         rx, ry = input_range[[0, 1], 0]
#         width, height = input_range[[0, 1], 1] - input_range[[0, 1], 0]

#         for obs in obstacles:
#             cx, cy = obs['x'], obs['y']
#             testX = torch.tensor(cx)
#             testY = torch.tensor(cy)

#             if (cx < rx):
#                 testX = rx
#             elif (cx > rx + width):
#                 testX = rx + width

#             if (cy < ry):
#                 testY = ry
#             elif (cy > ry + height):
#                 testY = ry + height
            
#             dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
#             if dist < obs['r']:
#                 return False
        
#         return True

# class QuadrotorConstraint(Constraint):
#     def is_safe(self, input_range):
#         delta = self.perturbation
#         yoffset1 = 1
#         zoffset1 = 3
#         yoffset2 = -1.5
#         zoffset2 = 1
#         little_radius = 1.25*0.4
#         big_radius = 0.03
#         yoffset3 = 0
#         zoffset3 = 0
#         obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
#                         {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
#                         {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
#                         {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
#                         {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
#                         {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
#                         {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
#                         {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
#                         {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
#                         {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
#                         {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
#                         {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},
#                         {'x': 0, 'y': -2. + yoffset3, 'r': little_radius},
#                         {'x': 0, 'y': 2. + yoffset3, 'r': little_radius},
#                         {'x': 0, 'z': -1. + zoffset3, 'r': little_radius},
#                         {'x': 0, 'z': 3. + zoffset3, 'r': little_radius}]
        
#         rx, ry, rz = input_range[[0, 1, 2], 0]
#         length, width, height = input_range[[0, 1, 2], 1] - input_range[[0, 1, 2], 0]

#         for obs in obstacles:
#             if 'y' in obs:
#                 cx, cy = obs['x'], obs['y']
#                 testX = torch.tensor(cx)
#                 testY = torch.tensor(cy)

#                 if (cx < rx):
#                     testX = rx
#                 elif (cx > rx + length):
#                     testX = rx + length

#                 if (cy < ry):
#                     testY = ry
#                 elif (cy > ry + width):
#                     testY = ry + width
                
#                 dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
#                 if dist < obs['r']:
#                     return False
#             if 'z' in obs:
#                 cx, cz = obs['x'], obs['z']
#                 testX = torch.tensor(cx)
#                 testZ = torch.tensor(cz)

#                 if (cx < rx):
#                     testX = rx
#                 elif (cx > rx + length):
#                     testX = rx + length

#                 if (cz < rz):
#                     testZ = rz
#                 elif (cz > rz + height):
#                     testZ = rz + height
                
#                 dist = torch.sqrt((cx-testX)**2 + (cz - testZ)**2)
#                 if dist < obs['r']:
#                     return False
        
#         return True