class UniversalAgent(object):
    def __init__(self, ECM, actions, adj_matrix):
        """
            Projective Simulation Agent. Any typical ECM can be used.

            Args:
                ECM (object): Episodic compositional memory (ECM). The brain of the agent.
                actions (np.ndarray): An array of possible actions. Specified by the environment.
                adj_matrix (np.ndarray): Adjancency matrix representing the structure of the default decision tree.
            """

        self.ECM = ECM
        self.adj_matrix = adj_matrix
        self.actions = actions

        # preprocessing the actions
        # storing the action clips in an dictionary: {action1 (str) : Action_clip1 (object), action2 (str) : Action_clip2 (object),...}
        # self.ECM.get_action_dict([str(action) for action in actions])

    def step(self, observation):
        """
        Given an observation, returns the id of an action clip.

        Args:
            observation (object): The observation in some form of encoding.
        Returns:
            actionClip_id (int): The id of the chosen action clip.
        """

        self.activeClip = self.ECM.get_or_create_percept_clip(observation, self.adj_matrix)
        actionClip_id = self.ECM.random_walk(self.activeClip)

        return actionClip_id

    def learn(self, reward):
        """
        Given a reward, updates adjancency matrix and g matrix.
        Args:
            reward (float): The received reward.
        """
        self.ECM.learn(reward)