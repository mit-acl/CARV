
class Analyzer:
    def __init__(self, cl_system, tf) -> None:
        self.reachable_sets = {}
        self.cl_system = cl_system
        self.tf = tf