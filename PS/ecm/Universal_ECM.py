from PS.ecm.Clips import ActionClip, PerceptClip
import numpy as np


class UniversalECM(object):
    def __init__(self, gamma_damping=0., eta_glow_damping=0.1, beta=1):
        """
        Basic episodic compositional memory (ECM) operating with any adjancency matrix which represents the clip
        structure of the PS network. The tree structure does not change while learning. This code could be extended with
        some generalization method, which adapt the decision tree while learning.

        Args:
            gamma_damping (float): The damping (or gamma) parameter. Set to zero if the environment doesn't change in time.
            eta_glow_damping (float): glow parameter. Defaults to 0.1.
            beta (float): softmax parameter. Defaults to 1.
        """

        self.num_actions = 0
        self.num_percepts = 0
        self.gamma_damping = gamma_damping
        self.eta_glow_damping = eta_glow_damping
        self.beta = beta

        self.percept_dict = {}  # stores the percept_clips with the corresponding percept, here: {percept (str) : Percept_clip (object), ...}
        self.action_dict = {}  # stores the action_clips with the corresponding number, here: {action (str) : Action_clip (object), ...}

    def get_action_dict(self, actions):
        """
        Initializes the action dictionary.
        It has this form: {action1 (str) : Action_clip1 (object), action2 (str) : Action_clip2 (object),...}

        Args:
            actions (array): [action1 (str), action2 (str), ...]
        """
        for action in actions:
            action_clip = ActionClip(self.num_actions, action)
            self.action_dict[action] = action_clip
            self.num_actions += 1

    def get_or_create_percept_clip(self, observation, adj_matrix):
        """
        Finds the corresponding percept clip to the observation or creates a new percept clip for the observation.
        (1) Preprocesses the observation to a percept.
        (2) Tries to find the corresponding percept clip to this percept; returns the clip.
        (3) Creates a new percept clip if the percept has not been encountered before.
        (4) Sets the decision tree (in form of the adjancency matrix) and the g matrix as attributes of the percept clip.

        Args:
            observation (object): The observation in some form of encoding.
            adj_matrix (np.ndarray): The adjancency matrix representing the decision tree
        Returns:
            percept_clip (object): percept clip
        """

        percept = self.preprocess(observation)

        if percept in self.percept_dict.keys():
            percept_clip = self.percept_dict[percept]
            return percept_clip
        else:
            # initialize g matrix with empty np.array
            g_matrix = np.zeros((len(adj_matrix), len(adj_matrix)))

            percept_clip = PerceptClip(self.num_percepts, percept, adj_matrix, g_matrix)
            self.percept_dict[percept] = percept_clip
            self.num_percepts += 1

            return percept_clip

    def get_percept_clip(self, percept):
        return self.percept_dict[percept]

    def get_action_clip(self, action):
        return self.action_dict[action]

    def random_step(self, from_clip_index, adj_matrix, g_matrix):
        """
        Does a random step (transition) from a clip (from_clip_index) to a connected clip (to_clip_index).
        The connected clips are weighted with the softmax distribution.

        Args:
            from_clip_index (int): from this clip a random step will be performed
            adj_matrix (np.ndarray): the adjancency matrix of the percept clip from which the random walk is performed
            g_matrix (np.ndarray): the  g matrix of the percept clip from which the random walk is performed
        Returns:
            to_clip_index (int): probabilistically picked clip
            finished (boolean): True, if an action clip is found
        """

        finished = False

        row = adj_matrix[from_clip_index]
        enum_row = list(enumerate(row))
        filtered_row = [enum_row[i] for i in range(len(enum_row)) if enum_row[i][1] != 0]

        if len(filtered_row) != 0:  # if the row has now entry != 0 an action clip is found

            # connected_clips_indices: indices of the connected clips
            # connected_clips_probabilities: h values of the connected clips
            connected_clips_indices = [filtered_row[i][0] for i in range(len(filtered_row))]
            connected_clips_probabilities = np.array([filtered_row[i][1] for i in range(len(filtered_row))])

            # pick one clip of all connected clips randomly weighted with the softmax distribution
            to_clip_index = np.random.choice(connected_clips_indices,
                                             p=self.softmax(self.beta, connected_clips_probabilities))

            # set g value of the clip transition to one
            x, y = from_clip_index, to_clip_index
            g_matrix[x, y] = 1

        else:
            to_clip_index = from_clip_index
            finished = True

        return to_clip_index, finished

    def random_walk(self, percept_clip):
        """
        Performs a random walk through the decision tree from the percept clip
        (encoded in the adjancency matrix) to an action clip. Returns the id (number) of this action clip.

        Args:
            percept_clip (object): percept clip
        Returns:
            action_clip_index (int): Id (number) of the found action clip
        """

        # check, if the input clip is a percept clip
        if percept_clip.type != "PerceptClip":
            print("first clip in random walk needs to be a percept_clip")

        # adjancency matrix g_matrix (np.ndarray) from the percept clip
        adj_matrix = percept_clip.adj_matrix
        g_matrix = percept_clip.g_matrix

        # all clips connected to a percept clip are in the first row
        from_clip_index = 0

        finished = False

        # do random steps until reaching a action clip
        while not finished:
            to_clip_index, finished = self.random_step(from_clip_index, adj_matrix, g_matrix)
            from_clip_index = to_clip_index

        # damp all g values from the whole network
        for clip in self.percept_dict.values():
            if clip != percept_clip:
                clip.g_matrix *= (1 - self.eta_glow_damping)

        # subtract one to get the action clip index, because the first clip is always the percept clip
        action_clip_index = to_clip_index - 1

        return action_clip_index

    def learn(self, reward):
        """
        Updates the h_values of all edges according to the reward.
        Sets all g values to zero.

        Args:
            reward (float): received reward
        """
        if reward != 0.0:
            for clip in self.percept_dict.values():
                clip.adj_matrix = clip.adj_matrix + reward * clip.g_matrix

            for clip in self.percept_dict.values():
                clip.g_matrix *= 0

    #### helper functions ####

    def softmax(self, beta, h_values):
        """
        Calculates probabilities according to the softmax distribution.

        Args:
            beta (float): softmax parameter beta
            h_values (np.ndarray): array of the h-values
        Returns:
            tuple of probabilities, same order as the h_values
        """
        ex = np.array(h_values, dtype=np.float64) * beta
        ex = ex - max(ex)
        ex = np.exp(ex)

        return ex / sum(ex)

    def preprocess(self, observation):
        """
        Given an observation, returns a percept.
        This function is just to emphasize the difference between observations
        issued by the environment and percepts which describe the observations
        as perceived by the agent.

        Args:
            observation (object): The observation in some form of encoding.
        Returns:
            percept (str): The observation encoded as a percept.
        """
        percept = str(observation)
        return percept
