class Clip():
    """a clip"""
    def __init__(self, id):
        self.id = id
        #self.label = self.id
        self.type = "Clip"


class ActionClip(Clip):
    """an action clip"""
    def __init__(self, id, action):
        Clip.__init__(self, id)
        self.label = action
        self.type = "ActionClip"


class PerceptClip(Clip):
    """a percept clip"""
    def __init__(self, id, percept, adj_matrix, g_matrix):
        Clip.__init__(self, id)
        self.label = percept
        self.adj_matrix = adj_matrix
        self.g_matrix = g_matrix
        self.type = "PerceptClip"