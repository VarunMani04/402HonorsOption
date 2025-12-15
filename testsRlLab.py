import matplotlib.pyplot as plt

from random import seed
from ParkingLotFactory import createParkingMDP
from agent.QLearningAgent import QLearningAgent
from MDP import MDP, generate_random_mdp
from agent.AgentBase import Verbosity
from statistics import stdev

def trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50):
    output = []
    for epoch in range(numEpochs):
        # first, learn for <numSamples> trajectories
        for trajectory in range(numSamples):
            # this function applies the learning rate scheduler
            #agent.setLearningRate(trajectory, numSamples)
            _ = mdp.simulateTrajectory(agent, startState=start)
            if agent.verbosity == Verbosity.VERBOSE:
                print("\n **************************Epoch {}, sample trial: {} Completed**************************".format(epoch + 1, trajectory + 1))

        # second, evaluate for <numSamples> trajectories
        agent.evaluating = True
        rewardList = []
        for _ in range(numSamples):
            reward = mdp.simulateTrajectory(agent, startState=start, MAX_ACTIONS=maxTrajectoryLen)
            rewardList.append(reward)
        agent.evaluating = False

        rewardStdev = 0
        if len(rewardList) > 1:
            rewardStdev = stdev(rewardList)
        # print to console
        print("At Epoch\t{}\tOn\t{}\t, Over {} Trajectories, Agent\t{}\t achieved a Grand Total Reward:\t{:.4f}\t, for an average of\t{:.4f}\t with std\t {:.4f}.".format(
                epoch, mdp.name, len(rewardList), agent.name, sum(rewardList), sum(rewardList) / len(rewardList),
                rewardStdev))
        output.append(sum(rewardList) / len(rewardList))

    print()
    return output


############ Task1: Understand updates on a Q-value table ###########
# In this task, we will apply the Q-learning on the simple MDP we saw in Lab1 (on slides and in MDP1.txt)
# Follow the detailed description in CANVAS to complete Task1
def RlLabTest1(seedValue):
    '''
    The Q-value table of all states with different actions will be printed for each sample trial in each example
    Average rewards of each trail will be reported
    This function will create the following files:
        BasicQLearner_Lab 1's toy MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Lab 1's toy MDP_rewardList.txt: Reward List of each epoch
    '''
    print("\n\n\n----------------------- BEGIN Test 1 - Q-Learning on the first toy MDP we saw in Lab 1 (the movie), with Seed " + str(seedValue))
    seed(seedValue)

    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")

    randomInit = False
    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, randomInit=randomInit, verbosity=Verbosity.VERBOSE)

    # set amount of training/testing to something small (so we arent flooded with output)
    start = 0
    numSamples = 1
    numEpochs = 1

    # see the policy before we do any observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    # see the policy after 2 trajectories worth of observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    #see the final policy after 4 trajectories worth of observations
    agent.printPolicy()

# same as last test, but with random table initialization
def RlLabTest2(seedValue):
    '''
    The Q-value table of all states with different actions will be printed for each sample trial in each example
    Average rewards of each trail will be reported
    This function will create the following files:
        BasicQLearner_Lab 1's toy MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Lab 1's toy MDP_rewardList.txt: Reward List of each epoch
    '''
    print("\n\n\n----------------------- BEGIN Test 2 - Q-Learning on the first toy MDP we saw in Lab 1 (the movie), but this time with random initialization, with Seed " + str(seedValue))
    seed(seedValue)

    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")

    randomInit = True
    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, randomInit=randomInit, verbosity=Verbosity.VERBOSE)

    # set amount of training/testing to something small (so we arent flooded with output)
    start = 0
    numSamples = 1
    numEpochs = 1

    # see the policy before we do any observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    # see the policy after 2 trajectories worth of observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    #see the final policy after 4 trajectories worth of observations
    agent.printPolicy()


############ **Task2: Creating learning curve of the Q-learning agent on the parking MDP** ###########
# Learning curve is a line chart which x-axis is the epoch and y-axis is the average rewords
# Your will run test2 and test3 to complete task 2 by following the detailed description in CANVAS
def RlLabTest3(seedValue):
    '''
    This is a test on the Small Parking MDP
    This function will create the following files:
        BasicQLearner_Small Parking MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Small Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Lab 3 Test 3 - Q-Learning on a Small Parking MDP, the movie, with Seed " + str(seedValue))

    # setup the mdp parameters
    mdp, start = createParkingMDP("tinyParkingLot", numRegularSpacesPerRow=1, numHandicappedSpacesPerRow=1)

    # set amount of training/testing to something small
    numSamples = 1
    numEpochs = 1
    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, randomInit=True, verbosity=Verbosity.VERBOSE)

    # see the policy before we do any observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    # see the policy after 2 trajectories worth of observations
    agent.printPolicy()
    trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)

    # see the final policy after 4 trajectories worth of observations
    agent.printPolicy()

#Task 3: Understand behavior of Q-learning agent on a random MDP:
def RlLabTest4(seedValue):
    '''
    This is a test on a hard Parking MDP
    Your are required to run test3 to collect the reward data for creating the learning curve
    This function will create the following files for you the generate the learning curve:
        BasicQLearner_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 4 - Q Learning agent on a Random MDP, with Seed " + str(
        seedValue))

    numStates = 40
    numActions = 20
    mdp = generate_random_mdp(numStates, numActions)
    mdp.name = "randomMDP"
    start = 0

    agent = QLearningAgent("BasicQLearner", mdp.numActions, mdp.numStates, randomInit=True)

    numSamples = 10
    numEpochs = 100
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
    #agent.printPolicy()
    print(agent.analyzeQfn())

    # plot the results
    plt.title("Rewards for a random MDP - " + mdp.name)
    plt.plot(range(numEpochs), rewards, label=agent.name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


############ ***Task4: Investigate the impacts of the hyperparameters of the Q-learning procedure ** ###########
# Your will run test4 and test 5 to complete task 3 by following the detailed instructions in Canvas
def RlLabTest5(seedValue):
    '''
    probGreedy in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - Comparing 3 (or more) Q learning agents, with Seed " + str(seedValue))

    mdp, start = createParkingMDP("busierParkingLot-.9", busyRate=.9)

    agent1 = QLearningAgent("GreedierQLearner-.999", mdp.numActions, mdp.numStates, probGreedy=.999)
    agent2 = QLearningAgent("LessGreedyQLearner-.5", mdp.numActions, mdp.numStates, probGreedy=.5)
    agent3 = QLearningAgent("MUCHLessGreedyQLearner-.001", mdp.numActions, mdp.numStates, probGreedy=.001)

    numSamples = 50
    numEpochs = 100
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
        allRewards.append(rewards)

        #agent.printPolicy()
        print(agent.analyzeQfn())

    # plot the results
    plt.title("Rewards varying the GREED of 3 agents")
    plt.plot(range(numEpochs), allRewards[0], 'r', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'b', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "g", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.show()


def RlLabTest6(seedValue):
    '''
    Learning rate in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 6 - Comparing Q learning agents in 3 (or more) different learning rates, with Seed " + str(seedValue))

    mdp, start = createParkingMDP("busierParkingLot", busyRate=.9)

    agent1 = QLearningAgent("HighLRQLearner-1", mdp.numActions, mdp.numStates, learningRate=1)
    agent2 = QLearningAgent("MiddleLRQLearner-.1", mdp.numActions, mdp.numStates, learningRate=.1)
    agent3 = QLearningAgent("LowLRQLearner-.0001", mdp.numActions, mdp.numStates, learningRate=.0001)

    numSamples = 50
    numEpochs = 100
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent)
        allRewards.append(rewards)

        #agent.printPolicy()
        print(agent.analyzeQfn())

    # plot the results
    plt.title("Rewards varying the LEARNING RATE of 3 agents")
    plt.plot(range(numEpochs), allRewards[0], 'r', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'b', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "g", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    plt.gca().set_ylim(bottom=0, top=200)
    plt.show()


############ ***Task5: Test the Q-learning procedure on different MDPs** ###########
def RlLabTest7(seedValue):
    '''
    Parking MDP with different busy rates will be created
    It will create the following files for you the generate the learning curve:
        BasicQLearner_Large Parking MDP{number}_rewardCurveData.txt: Average reward of each epoch
        BasicQLearner_Large Parking MDP{number}_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 7 - Comparing 3 (or more) MDPs, with Seed " + str(seedValue))

    # all the parking lots are the same size and thus have the same start position
    mdp1, start = createParkingMDP("VERYbusyParkingLot-.999", busyRate=.999)
    mdp2, _ = createParkingMDP("halfBusyParkingLot-.5", busyRate=.5)
    mdp3, _ = createParkingMDP("mostlyEmptyParkingLot-.1", busyRate=.1)

    numSamples = 50
    numEpochs = 100
    mdps = [mdp1, mdp2, mdp3]
    allRewards = []
    for i in range(len(mdps)):
        agent = QLearningAgent("BasicQLearner", mdps[i].numActions, mdps[i].numStates)
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent)
        allRewards.append(rewards)

        #agent.printPolicy()
        print(agent.analyzeQfn())

    # plot the results
    plt.title("Rewards varying the BUSYNESS of 3 parking MDPs")
    plt.plot(range(numEpochs), allRewards[0], 'r', label=mdps[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'b', label=mdps[1].name)
    plt.plot(range(numEpochs), allRewards[2], "g", label=mdps[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.show()

#Task 6: Leading toward Line and Grid Search:
def RlLabTest8(seedValue):
    '''
    Parking MDP with different busy rates will be created
    For each MDP, Q-learning with three settings of probGreedy will be testied
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP{number}_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP{number}_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 8 - Comprehensively evaluate Q-learning parameters, with Seed " + str(seedValue))

    # make the same 3 MDPs as in the last test
    # all the parking lots are the same size and thus have the same start position
    mdp1, start = createParkingMDP("VERYbusyParkingLot-.99", busyRate=.99)
    mdp2, _ = createParkingMDP("halfBusyParkingLot-.5", busyRate=.5)
    mdp3, _ = createParkingMDP("mostlyEmptyParkingLot-.2", busyRate=.2)

    numSamples = 50
    numEpochs = 100
    mdps = [mdp1, mdp2, mdp3]
    for i in range(len(mdps)):
        agent1 = QLearningAgent("GreedierQLearner", mdps[i].numActions, mdps[i].numStates, probGreedy=.99999)
        rewards1 = trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent1)

        agent2 = QLearningAgent("LessGreedyQLearner", mdps[i].numActions, mdps[i].numStates, probGreedy=.7)
        rewards2 = trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent2)

        agent3 = QLearningAgent("MUCHLessGreedyQLearner", mdps[i].numActions, mdps[i].numStates, probGreedy=.1)
        rewards3 = trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent3)

        # plot the results
        plt.title("Rewards varying GREEDINESS on MDP - " + mdps[i].name)
        plt.plot(range(numEpochs), rewards1, 'r', label=agent1.name)
        plt.plot(range(numEpochs), rewards2, 'b', label=agent2.name)
        plt.plot(range(numEpochs), rewards3, "g", label=agent3.name)
        plt.ylabel('Rewards')
        plt.xlabel('Epoch')
        plt.legend()
        plt.gca().set_ylim(bottom=0)
        plt.show()
