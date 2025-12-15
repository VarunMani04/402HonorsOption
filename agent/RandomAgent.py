from agent.AgentBase import AgentBase
from random import randrange

# This agent simply behaves randomly (within the rules).
class RandomAgent(AgentBase):
    def __init__(self, numActions, verbosity):
        super().__init__("RandomAgent", verbosity)
        self.numActions = numActions

    def selectAction(self, state, iteration, mdp):
        return randrange(self.numActions)
