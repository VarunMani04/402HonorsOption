from random import random

class MDP():
    def __init__(self, numStates, numActions, name, filename=None):
        self.name = name
        if not filename: # create a blank MDP with all 0, based on specified sizes
            self.numStates = numStates
            self.numActions = numActions

            # this is now to be indexed [ACTION][STATE_FROM][STATE_TO] and returns a probability.
            # Python doesn't really do 2d arrays, but nested lists are close enough.
            self.transitionFn = [[[0 for _ in range(self.numStates)] for _ in range(self.numStates)] for _ in range(self.numActions)]

            # this is now to be indexed [state] and returns a reward
            self.rewardFn = [0 for _ in range(self.numStates)]

            # this is now to be indexed [state] and returns a boolean, indicating whether that state is terminal or not
            self.isTerminal = [False for _ in range(self.numStates)]
        else:
            self.readFromFile(filename)

    def simulateTrajectory(self, agent, startState, MAX_ACTIONS=20):
        totalReward = 0
        currentState = startState

        for i in range(MAX_ACTIONS):
            rewardHere = self.rewardFn[currentState]
            totalReward += rewardHere

            action = agent.selectAction(currentState, i, self)
            nextState = self.computeNextState(currentState, action)
            agent.observeReward(i, currentState, nextState, action, totalReward, rewardHere)

            if self.isTerminal[currentState]:
                break
            currentState = nextState

        agent.endEpisode(i, currentState, totalReward)
        return totalReward

    def computeNextState(self, state, action):
        randProb = random()
        cumulativeProb = 0

        for i in range(self.numStates):
            probThisState = self.transitionFn[action][state][i]
            cumulativeProb += probThisState
            if randProb < cumulativeProb:
                return i
        print("+++++++ problem found in computeNextState\n")
        return -1

    def readFromFile(self, filename):
        file = open(filename, "r", encoding="utf-8-sig")
        lines = file.readlines()
        file.close()

        # first line of the file has state/action size info
        line1 = lines[0].split()
        self.numStates = int(line1[0])
        self.numActions = int(line1[1])

        # second line is blank, so we start reading TRANSITION FUNCTION on the third line
        lineIdx = 2
        self.transitionFn = []
        for i in range(self.numActions):
            matrixForThisAction = []
            for j in range(self.numStates):
                splitLine = lines[lineIdx + j].split()
                for i in range(len(splitLine)):
                    splitLine[i] = float(splitLine[i])
                matrixForThisAction.append(splitLine)
            self.transitionFn.append(matrixForThisAction)
            lineIdx += self.numStates + 1

        # next block is the reward function
        self.rewardFn = lines[lineIdx].split()
        for i in range(len(self.rewardFn)):
            self.rewardFn[i] = float(self.rewardFn[i])
        lineIdx += 2

        # last block is the terminal specification
        self.isTerminal = lines[lineIdx].split()
        for i in range(len(self.isTerminal)):
            if self.isTerminal[i] == "1":
                self.isTerminal[i] = True
            else:
                self.isTerminal[i] = False

    # this function handles calls to print()
    def __repr__(self):
        result = self.name + "\n"
        result += "numStates " + str(self.numStates) + " numActions " + str(self.numActions) + "\n\n"

        # print the transition function
        for k in range(self.numActions):
            result += "Action " + str(k) + " Transition Function\n"
            for i in range(self.numStates):
                for j in range(self.numStates):
                    result += str(self.transitionFn[k][i][j]) + "\t"
                result += "\n"
            result += "\n"

        result += "Reward function\n"
        # print the transition function
        for i in range(self.numStates):
            result += str(self.rewardFn[i]) + "\t"
        result += "\n\n"

        result += "isTerminal\n"
        # print the terminal list
        for i in range(self.numStates):
            result += str(self.isTerminal[i]) + "\t"
        return result

    # this function handles calls to str()
    def __str__(self):
        return self.__repr__()

def generate_random_mdp(numStates, numActions):
    '''
    It creates a random MDP for the given number of states and actions.
    The transition probabilities for each state-action pair are randomly distributed across all next states
    The rewards are picked randomly between -10 and 10
    Terminal states are chosen with  0.1 probability.
    '''
    mdp = MDP(numStates, numActions, "randomMDP")

    # Randomly initialize transition function
    for action in range(numActions):
        for state in range(numStates):
            remaining_prob = 1.0
            for next_state in range(numStates - 1):
                trans_prob = round(random() * remaining_prob, 2)
                mdp.transitionFn[action][state][next_state] = trans_prob
                remaining_prob -= trans_prob
            mdp.transitionFn[action][state][numStates - 1] = remaining_prob

    # Randomly initialize reward function between -10 and 10
    for state in range(numStates):
        mdp.rewardFn[state] = round((random() * 20) - 10, 2)  # random number between -10 and 10

    # Randomly initialize isTerminal
    for state in range(numStates):
        mdp.isTerminal[state] = random() < 0.1

    return mdp
