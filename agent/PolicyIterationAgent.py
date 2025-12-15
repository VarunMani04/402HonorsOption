from random import randrange
from agent.AgentBase import AgentBase, Verbosity

UNINITIALIZED_VALUE = 0 # Here we initialize with 0s. A different constant or a random initialization would probably be better, but this works

class StationaryValueFn():
    def __init__(self, numStates):
        self.numStates = numStates
        # this table is indexed by STATES and contains a VALUE
        self.theFn = [UNINITIALIZED_VALUE for _ in range(self.numStates)]

    def elementWiseSubtraction(self, rhs):
        result = StationaryValueFn(self.numStates)
        for i in range(self.numStates):
            result.theFn[i] = self.theFn[i] - rhs.theFn[i]
        return result
    
    def maxNorm(self):
        '''
        Returns the maximum value in the value function
        '''
        maxVal = self.theFn[0]
        for i in range(self.numStates):
            if self.theFn[i] > maxVal:
                maxVal = self.theFn[i]
        return maxVal

    def computeActionEV(self, currentState, action, MDP):
        '''
        # computeActionEV calculates the expected value of an action from a specific state using the formula:
        # ExpectedValue(s, a) = SUM(P(s' | s, a) * V(s')) over all s'
        # where:
        # s is the current state
        # a is the chosen action
        # s' represents possible next states after taking action a from state s
        # P(s' | s, a) is the transition probability to state s' when action a is taken in state s
        # V(s') is the value of state s'
        '''
        
        result = 0
        for nextState in range(self.numStates):
            probability = MDP.transitionFn[action][currentState][nextState]
            value = self.theFn[nextState]
            result += probability*value
        return result

    def bellmanBackup(self, discountFactor, MDP):
        result = StationaryValueFn(self.numStates)
        for state in range(self.numStates):
            actionValues = [self.computeActionEV(state, action, MDP) for action in range(MDP.numActions)]
            result.theFn[state] = MDP.rewardFn[state] + discountFactor * max(actionValues)
        return result

    def __repr__(self):
        result = "Stationary VALUE function (How good agent the agent thinks it is to be in each state)\n"
        for state in range(self.numStates):
            result += "State " + str(state) + ":\t" + str(self.theFn[state]) + "\n"
        return result

class StationaryPolicy():
    def __init__(self, numStates):
        self.numStates = numStates
        # this table is indexed by STATES and contains an ACTION
        self.policy = [UNINITIALIZED_VALUE for _ in range(self.numStates)]

    def evaluate(self, discountFactor, epsilon, MDP, maxIterations=200):
        thisIterVFN = StationaryValueFn(self.numStates)
        nextIterVFN = StationaryValueFn(self.numStates)

        for i in range(maxIterations):
            nextIterVFN = thisIterVFN.bellmanBackup(discountFactor, MDP)
            difference = nextIterVFN.elementWiseSubtraction(thisIterVFN)
            if epsilon >= difference.maxNorm():
                break
            thisIterVFN = nextIterVFN
        return nextIterVFN

    def __repr__(self):
        result = "Stationary POLICY (Action the agent will take in each state)\n"
        for state in range(self.numStates):
            result += "State " +str(state) + ":\t" + chr(self.policy[state] + ord('A')) + "\n"
        return result

# -----------------------------------------------------------
class PolicyIterationAgent(AgentBase):
    def __init__(self, name, numStates, verbosity):
        super().__init__(name, verbosity)
        self.numStates = numStates

        # this table is indexed by STATES and contains an ACTION
        self.thePolicy = StationaryPolicy(numStates)
    
    # Given a state, get the action from the policy
    def selectAction(self, state, iteration, mdp):
        return self.thePolicy.policy[state]

    def randomizePolicy(self, numActions):
        self.thePolicy.policy = [randrange(numActions) for _ in range(self.numStates)]

    def solvePolicyIteration(self,  mdp, discountFactor=.7, epsilon=.001, maxIterations=200, verbosity=Verbosity.SILENT):
        '''
        Implements the policy iteration algorithm to solve MDP

        The method alternates between two steps until convergence or reaching maxIterations:
        1. Policy Evaluation: Compute the value function for the current policy.
        - This step finds out how good the current policy is by calculating the expected value of each state.
        2. Policy Improvement: Update the policy based on the computed value function.
        - For each state, this step checks all possible actions and selects the action that provides the highest expected return.

        Parameters:
        - discountFactor: 
            The discount factor used to discount future rewards, a value between 0 and 1.
        - epsilon: 
            The threshold of minimal updates for convergence of policy evaluation. 
        - numActions: 
            Total number of possible actions in the MDP.
        - mdp: 
            The Markov Decision Process to be solved.
        - verbosity: 
            Determines the level of logs to be printed during the execution.
        - maxIterations (default=200): 
            The maximum number of policy iteration cycles allowed before the method terminates.
        '''


        if verbosity == Verbosity.VERBOSE:
            print("MDP Diagnostic Info")
            print(mdp)
            print("\nStarting Policy iteration - discountFactor = ", str(discountFactor), " epsilon = ", str(epsilon))

        for iter in range(maxIterations):
            actionImproved = False
            currentValueFn = self.thePolicy.evaluate(discountFactor, epsilon, mdp)

            if verbosity == Verbosity.VERBOSE:
                print("\n***** Starting iteration ", iter)
                print(self.thePolicy)
                print(currentValueFn)

            for state in range(self.numStates):
                actionValues = []
                for action in range(mdp.numActions):
                    actionValue = currentValueFn.computeActionEV(state, action, mdp)
                    actionValues.append(actionValue)

                currentActionInThisState = self.thePolicy.policy[state]

                # If a better action is found, update the policy
                for action in range(mdp.numActions):
                    if actionValues[action] > actionValues[currentActionInThisState]:
                        if verbosity == Verbosity.VERBOSE:
                            print("Setting state ", str(state), "'s action to ", action, " from ", currentActionInThisState, ". ActionValues: ", actionValues)
                        currentActionInThisState = action
                        actionImproved = True
                self.thePolicy.policy[state] = currentActionInThisState
            
            # If no improvement, stop the policy iteration
            if not actionImproved:
                if verbosity == Verbosity.VERBOSE:
                    print("******* Function complete, final policy after ", str(iter), " iterations")
                    print(self.thePolicy)
                    print(currentValueFn)
                    print()
                return iter
        return maxIterations

    def __repr__(self):
        return str(self.thePolicy)
