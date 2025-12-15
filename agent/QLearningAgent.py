from random import random, randrange
from agent.AgentBase import AgentBase, Verbosity
from ParkingDefs import StateType, Act # only needed for the qFn interpretation

UNINITIALIZED_VALUE = 0 # Here we initialize with a small negative number (so that when we observe 0 reward things change)

class QLearningAgent(AgentBase):
    def __init__(self, name, numActions, numStates, probGreedy=.8, discountFactor=.7, learningRate=.1, randomInit=True, verbosity=Verbosity.SILENT):
        super().__init__(name, verbosity)
        self.probGreedy = probGreedy
        self.evaluating = False
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.numActions = numActions
        self.numStates = numStates

        if (randomInit):
            self.setupRandomQfunction()
        else:
            self.setupClearQfunction()

    def setupRandomQfunction(self):
        # this is a table to be indexed [STATE][ACTION] and contains a VALUE. Python doesnt really do 2d arrays, but nested lists are close enough.
        self.qFn = [[randrange(-50, 50) for _ in range(self.numActions)] for _ in range(self.numStates)]

    def setupClearQfunction(self):
        # this is a table to be indexed [STATE][ACTION] and contains a VALUE. Python doesnt really do 2d arrays, but nested lists are close enough.
        self.qFn = [[UNINITIALIZED_VALUE for _ in range(self.numActions)] for _ in range(self.numStates)]

    # returns the argmax across actions, given a state
    def findHighestValuedAction(self, state):
        bestValue = self.qFn[state][0]  # initialize to the first action
        bestAction = 0  # because we initialized to first element
        for action in range(1, self.numActions):  # can skip first element due to initialization
            valueHere = self.qFn[state][action]
            if valueHere > bestValue:
                bestValue = valueHere
                bestAction = action
        return bestAction

    def selectAction(self, state, iteration, mdp):
        if self.evaluating or random() < self.probGreedy: # this is the greedy choice, according to our current value function
            return self.findHighestValuedAction(state)
        else: # take a random action!
            return randrange(self.numActions)

    def observeReward(self, iteration, currentState, nextState, action, totalReward, rewardHere):
        if not self.evaluating:  # Perform the Q function update.
            if self.verbosity == Verbosity.VERBOSE:
                print(self) # this prints the Q-table
            # We begin by determining what action we will take once arriving in the next state so we can get a value from our table
            bestNextAction = self.findHighestValuedAction(nextState)

            # Now that we know the action we take next, calculate the delta using the observed reward, discount factor, and the contents of the table
            qDeltaHere = rewardHere + self.discountFactor * (self.qFn[nextState][bestNextAction] - self.qFn[currentState][action])

            # ... and at last, we update the table based on delta and learning rate
            self.qFn[currentState][action] += self.learningRate * qDeltaHere
        # This is a tricky way to ensure that whatever the parent class's overridden function was supposed to do actually happens (in this case it is print, based on verbosity level)
        return super().observeReward(iteration, currentState, nextState, action, totalReward, rewardHere)

    def setLearningRate(self, iteration, totalIterations):
        self.learningRate = .1 + (totalIterations - iteration) / totalIterations

    def printPolicy(self):
        print("state\tAction the agent prefers in that state")
        for state in range(self.numStates):
            action = self.findHighestValuedAction(state)
            print(str(state) + ": \t" + chr(action + ord('A')))

    # This function is peculiar to the parking MDP domain, everything else provided here is general to table-based Q-learning
    def analyzeQfn(self):
        numSpacesDeclined = 0
        for state in range(self.numStates):
            stateClass = StateType.get(state)
            if StateType.DRIVING_AVAILABLE == stateClass:
                if self.qFn[state][Act.PARK.value] < self.qFn[state][Act.DRIVE.value]:
                    numSpacesDeclined += 1
        result = self.name + "'s final Q function declines to park in unoccupied spaces " + str(numSpacesDeclined) + " times\n"
        return result

    def __repr__(self):
        result = "\nLearning Rate: " + str(self.learningRate) + "\n\t"

        for i in range(self.numActions):
            result += 'Action{:>2}'.format(chr(i + ord('A'))) + "\t"
        result += "\n"

        # transposing for printing purposes since action space < state space
        for i in range(self.numStates):
            result += str(i) + ":\t"
            for j in range(self.numActions):
                theValue = self.qFn[i][j]
                if UNINITIALIZED_VALUE == theValue:
                    result += "--------\t"
                else:
                    formattedNum = "{: 7.1f}".format(theValue)
                    if len(formattedNum) > 8: # string length is equal to the length of the scientific notation formatting string on next lines
                        formattedNum = "{:.1E}".format(theValue)
                    result += '{:>8}'.format(formattedNum) + "\t"
            result += "\n"
        return result

