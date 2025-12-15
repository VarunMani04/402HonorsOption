import abc
from enum import Enum

class Verbosity(Enum):
    VERBOSE = 0
    REWARDS = 1
    RESULTS = 2
    SILENT = 3

# This is the abstract base class (hence the use of the abc import) that specifies an agent
class AgentBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, verbosity):
        self.name = name
        self.verbosity = verbosity

    # This specifies that all Agents MUST specify a move function to override this one
    @abc.abstractmethod
    def selectAction(self, state, iteration, mdp):  pass

    # this MAY be overridden, but the base class provides some helpful output behavior and determines when an episode ends
    def observeReward(self, iteration, currentState, nextState, action, totalReward, rewardHere):
        if self.verbosity.value <= Verbosity.REWARDS.value:
            print("*************************")
            print("t : " + str(iteration) + "   Total Reward : \t" + str(totalReward))
            print("Action       : \t" + chr(action + ord('A')))
            print("Reward       : \t", rewardHere)
            print("Current State: \t" + str(currentState))

    # This MAY be overridden, but the base class provides some helpful output behavior ONLY
    def endEpisode(self, iteration, finalState, totalReward):
        if self.verbosity.value <= Verbosity.RESULTS.value:
            print("***************************************************************************")
            print("Iterations used       : \t" + str(iteration + 1))
            print("Final State: \t" + str(finalState))
            print("!!!Reward for this trial: \t", totalReward)
