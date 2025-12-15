import matplotlib.pyplot as plt
from random import seed
from ParkingLotFactory import createParkingMDP
from agent.PolicyIterationAgent import PolicyIterationAgent
from agent.QLearningAgent import QLearningAgent
#from agent.ProbPolicyAgents import OccupiedRandomNoHandicapLapAgent
from agent.AgentBase import Verbosity
from statistics import stdev
from MDP import MDP, generate_random_mdp

'''
numRows = 2
numRegularSpacesPerRow = 10
numHandicappedSpacesPerRow = 5
busyRate = .3
handicapBusyRate = .05
parkedReward = 1000
crashPenalty = -10000
waitingPenalty = -1
decayBusyRate = False
decayReward = True

numActions = 2
parkProb = .5
numSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow
start = 4 * numSpacesPerRow - 2 # Be careful to update this variable and the previous one if you change the parking lot size
discountFactor = .7
epsilon = .001
'''

def measureAgentPerformance(numSamples, agent, mdp, startState):
    rewardList = []
    for _ in range(numSamples):
        reward = mdp.simulateTrajectory(agent, startState=startState)
        rewardList.append(reward)
    print("On\t{}\t, Over {} Trajectories, Agent\t{}\t achieved a Grand Total Reward:\t{:.4f}\t, for an average of\t{:.4f}\t with std\t {:.4f}\t.".format(
            mdp.name, len(rewardList), agent.name, sum(rewardList), sum(rewardList) / len(rewardList),
            stdev(rewardList)))
    return rewardList
    # Uncomment the below lines if you want to print the list and not stick it in a file
    # for i in range(len(rewardList)):
    #     print(rewardList[i], end="\t")
    # print()

def trainQagent(numEpochs, numSamples, mdp, start, agent):
    '''
    This is the function to train the Q-learning table
    '''
    for epoch in range(numEpochs):
        for trajectory in range(numSamples):
            agent.setLearningRate(trajectory, numSamples)
            _ = mdp.simulateTrajectory(agent, startState=start)


############ Task1: Understand policy iteration ###########
# A. Investigate PolicyIterationAgent.py's implementation to understand how policy is stored
# and updated during the policy iteration.
# B. Run lab6test1 and lab6test2 to check the updates of Stationary POLICY (Action the agent will take in each state)
# C. (TURN THIS IN) Compare and contrast the final policy and initial random policy for both MDPs, and provide a written iterpretation

def PolicyIterLabTest1(seedValue):
    '''
    Executes the policy iteration algorithm on the MDP specified in "data/MDP1.txt".
    The following information will be output:
        1. MDP info: Transtion Function, Reward function, Terminal states
        2. Policy interation info: Stationary POLICY, Stationary VALUE function
    '''

    print("\n\n\n----------------------- BEGIN Test 1 - See 'Policy iteration: the Movie' on the simple MDP1, with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp
    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")

    # create an agent that always takes the 0th action
    agent = PolicyIterationAgent("PolicyIterationAgent", mdp.numStates, Verbosity.VERBOSE)

    # make that agent have a random policy
    agent.randomizePolicy(mdp.numActions)

    # perform policyIteration
    agent.solvePolicyIteration(mdp, verbosity=Verbosity.VERBOSE)

def PolicyIterLabTest2(seedValue):
    print("\n\n\n----------------------- BEGIN Test 2 - See 'Policy iteration: the Movie' on the simple MDP2, with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp
    mdp = MDP(None, None, "MDP2", "data/MDP2.txt")

    # create an agent that always takes the 0th action
    agent = PolicyIterationAgent("PolicyIterationAgent", mdp.numStates, Verbosity.VERBOSE)

    # make that agent have a random policy
    agent.randomizePolicy(mdp.numActions)

    # perform policyIteration
    agent.solvePolicyIteration(mdp, verbosity=Verbosity.VERBOSE)

############ Task2: Measure policy iteration on MDP2   ###########
# A. Investigate the measureAgentPerformance() function in this file to understand how to evaluate a policy
# B. Run lab6test3 to see how well the PolicyIterationAgent performs on a simple MDP
# C. (TURN THIS IN) Draw a chart to compare the performance of the initial random policy with the final learned policy.
# D. (TURN THIS IN) Interpret your chart: How does the initial random policy perform relative to the final policy

def PolicyIterLabTest3(seedValue):
    '''
    Executes the policy iteration algorithm on the MDP specified in "data/MDP2.txt" and measures the performance of
    the initial random policy and the learned policy on "data/MDP2.txt".

    The function performs the following major steps:
        1. Creates a Policy Iteration Agent.
        2. Initializes the agent's policy to a random policy and then measures its performance.
        3. The agent then performs policy iteration to refine the policy.
        4. After learning, the agent's performance with the new policy is again measured.
        
    Relevant policies and performance metrics are printed for analysis.
    '''

    print("\n\n\n----------------------- BEGIN Test 3 - Measure quality of policy iteration on MDP2, with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp
    start = 0
    mdp = MDP(None, None, "MDP2", "data/MDP2.txt")

    # create an agent that always takes the 0th action
    agent = PolicyIterationAgent("PolicyIteration", mdp.numStates, Verbosity.SILENT)

    # make that agent have a random policy and then print it
    agent.randomizePolicy(mdp.numActions)
    print("RANDOM POLICY")
    print(agent.thePolicy)

    # measure the performance of the policy
    numSamples = 500
    preRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # perform policyIteration and then print the resulting policy
    iterations = agent.solvePolicyIteration(mdp)
    print("LEARNED POLICY (after ", iterations, " iterations)")
    print(agent.thePolicy)

    # measure the performance of the policy
    postRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticklabels(["Before", "After"])
    plt.title("Policy Iteration: Before and after")
    plt.boxplot([preRewards, postRewards])
    plt.ylabel('Rewards')
    plt.show()

############ Task3: Test Policy Iteration on different MDPs  ###########
# A. Run test4 to see how policy iteration performs on a simple parking MDP
# B. Run test5 to see how policy iteration performs on a hard parking MDP
# C. (TURN THIS IN) Draw a chart to compare the performance of final learned policies on two different parking MDPs.
# D. (TURN THIS IN) Interpret your chart. How big is the performance gap? Why do you think it exists?

def PolicyIterLabTest4(seedValue):
    '''
    This is the test on a simpler parking MDP (low bustRate)
    '''
    print("\n\n\n----------------------- BEGIN Test 4 - Policy iteration on simple parking MDP, with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp parameters
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .3
    mdp, start = createParkingMDP("mostlyEmptyParkingLot", numRegularSpacesPerRow=numRegularSpacesPerRow, numHandicappedSpacesPerRow=numHandicappedSpacesPerRow, busyRate=busyRate)

    # create an agent that always takes the 0th action
    agent = PolicyIterationAgent("PolicyIteration", mdp.numStates, Verbosity.SILENT)

    # make that agent have a random policy and then print it
    agent.randomizePolicy(mdp.numActions)
    print("RANDOM POLICY")
    print(agent.thePolicy)

    # measure the performance of the policy
    numSamples = 500
    preRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # perform policyIteration and then print the resulting policy
    iterations = agent.solvePolicyIteration(mdp)
    print("LEARNED POLICY (after ", iterations, " iterations)")
    print(agent.thePolicy)

    # measure the performance of the policy
    postRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    print(postRewards)

    # plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticklabels(["Before", "After"])
    plt.title("Policy Iteration: Before and after")
    plt.boxplot([preRewards, postRewards])
    plt.ylabel('Rewards')
    plt.show()

def PolicyIterLabTest5(seedValue):
    '''
    This is the test on a harder parking MDP (high busyRate)
    '''
    print("\n\n\n----------------------- BEGIN Test 5 - Policy iteration on harder parking MDP, with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp parameters
    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate)

    # create an agent that always takes the 0th action
    agent = PolicyIterationAgent("PolicyIteration", mdp.numStates, Verbosity.SILENT)

    # make that agent have a random policy and then print it
    agent.randomizePolicy(mdp.numActions)
    print("RANDOM POLICY")
    print(agent.thePolicy)

    # measure the performance of the policy
    numSamples = 500
    preRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # perform policyIteration and then print the resulting policy
    iterations = agent.solvePolicyIteration(mdp)
    print("LEARNED POLICY (after ", iterations, " iterations)")
    print(agent.thePolicy)

    # measure the performance of the policy
    postRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticklabels(["Before", "After"])
    plt.title("Policy Iteration: Before and after")
    plt.boxplot([preRewards, postRewards])
    plt.ylabel('Rewards')
    #plt.gca().set_ylim(bottom=60, top=100)  # you may find it helpful to restrict the Y axis if you want to focus on some part of output

    plt.show()

def PolicyIterLabTest6(seedValue):
    '''
    Executes Q-learning and a probabilistic policy on a parking MDP 

    This function performs the following major steps:
        1. Q-learning Agent Setup and Training
        2. Probabilistic Policy Setup and Evaluation
    Outputs:
        - Printed details of the Q-learning agent's and the probabilistic agent's policies and performance metrics.
    '''

    print("\n\n\n----------------------- BEGIN Test 6 - Tests different agents on parking MDP , with Seed " + str(seedValue))
    seed(seedValue)
    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate)

    numSamples = 500

    # setup the Q-learning agent
    learningRate = .1
    probGreedy = .9
    # set up a Q-learning Agent 
    Qagent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, probGreedy=probGreedy, learningRate=learningRate)

    # train the Q-learning agent
    numEpochs = 50
    trainQagent(numEpochs, numSamples, mdp, start, Qagent)
    Qagent.evaluating = True
    # Evaluation the trained Q-learning agent
    print("\nResult on Q-learning")
    viRewards = measureAgentPerformance(numSamples, Qagent, mdp, start)

    # create a PolicyIterationAgent
    agent = PolicyIterationAgent("PolicyIteration", mdp.numStates, Verbosity.SILENT)
    agent.randomizePolicy(mdp.numActions)
    # perform policyIteration and then print the resulting policy
    iterations = agent.solvePolicyIteration(mdp)
    # measure the performance of the policy
    print("\nResult on PolicyIteration after ", iterations, " iterations")
    piRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticklabels(["Q-Learning", "Policy Iteration"])
    plt.title("Q Learning vs Policy Iteration")
    plt.boxplot([viRewards, piRewards])
    plt.ylabel('Rewards')
    plt.show()




############ Task5: Compare three agent types on a RANDOM MDP: Probabilistic Policies, QLearningAgent and PolicyIterationAgent on a randomly generated MDP  ###########
# A. Run test7 to see results for each agent on the parking MDP
# B. (TURN THIS IN) Draw a chart to compare the rewards of Policy iteration, Qlearning agent, and Probabilistic Policy the parking MDP.
# C. (TURN THIS IN) Provide a written interpretation of what you see in the chart.
# D. (TURN THIS IN) While showing your work, compute the size of the Q-value table required for this MDP.
# E. (TURN THIS IN) Next, compute the number of updates this training loop will perform on the Q-value table.
# F. (TURN THIS IN) Explain why you think this isn't enough training.
# G. (TURN THIS IN) Describe how you would like to adjust the parameters WILL be enough and rerun the test.
# H. (TURN THIS IN) Characterize how your adjustment impacted the results and repeat until you are satisfied.

def PolicyIterLabTest7(seedValue):
    '''
    Executes Policy iteration, Q-learning and a probabilistic policy on a random hard mdp
    '''
    print("\n\n\n----------------------- BEGIN Test 7 - Tests of different agents on random hard mdp, with Seed " + str(seedValue))
    seed(seedValue)

    print("Note: this might take some time.")
    numSamples = 50
    numEpochs = 50  # Here is the line you should be adjusting for Lab 4's component of Assignment 1
    numStates = 200
    numActions = 100
    start = 0

    # generate a random MDP that is pretty big
    mdp = generate_random_mdp(numStates, numActions)

    # create a PolicyIterationAgent
    agent = PolicyIterationAgent("PolicyIteration", mdp.numStates, Verbosity.SILENT)
    agent.randomizePolicy(mdp.numActions)

    print("Performing policy iteration...")
    iterations = agent.solvePolicyIteration(mdp)

    print("\nResult on PolicyIteration after ", iterations, " iterations")
    piRewards = measureAgentPerformance(numSamples, agent, mdp, start)

    # create a QLearningAgent
    learningRate = .1
    probGreedy = .9
    Qagent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, probGreedy=probGreedy, learningRate=learningRate)

    print("Training QLearning Agent...")
    trainQagent(numEpochs, numSamples, mdp, start, Qagent)
    Qagent.evaluating = True

    print("\nResult on Q-learning")
    viRewards = measureAgentPerformance(numSamples, Qagent, mdp, start)

    # plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticklabels(["Q-Learning", "Policy Iteration"])
    plt.title("Q Learning vs Policy Iteration")
    plt.boxplot([viRewards, piRewards])
    plt.ylabel('Rewards')
    plt.show()

