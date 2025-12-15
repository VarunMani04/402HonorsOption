import matplotlib.pyplot as plt

from random import seed
from ParkingLotFactory import createParkingMDP
from agent.DeepQNetworkAgent import DeepQNetworkAgent, Verbosity
from MDP import MDP, generate_random_mdp
from statistics import stdev

def trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50):
    output = []
    for epoch in range(numEpochs):
        # first, learn for <numSamples> trajectories
        for trajectory in range(numSamples):
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

def DqnLabTest1(seedValue):
    print("\n\n\n----------------------- BEGIN Test 1 - DQN on the first toy MDP we saw in Lab 1 (the movie), with Seed " + str(seedValue))
    seed(seedValue)

    # setup the mdp parameters
    start = 0
    mdp = MDP(None, None, "MDP1", "data/MDP1.txt")

    # set amount of training/testing to something small
    numSamples = 2
    numEpochs = 1

    # configure the DQN
    layer_sizes = [128] # make a 1 layer MLP with 128 neurons, this is the default
    regularization = 0 # no regularization, parameters can be as large as they wish
    agent = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue, verbosity=Verbosity.VERBOSE)
    
    _ = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)

    print(agent)

def DqnLabTest2(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Lab 3 Test 2 - DQN on a Small Parking MDP, the movie, with Seed " + str(seedValue))

    # setup the mdp parameters
    numRegularSpacesPerRow = 1
    numHandicappedSpacesPerRow = 1
    busyRate = .3
    mdp, start = createParkingMDP("mostlyEmptyParkingLot", numRegularSpacesPerRow=numRegularSpacesPerRow, numHandicappedSpacesPerRow=numHandicappedSpacesPerRow, busyRate=busyRate)

    # set amount of training/testing to something small
    numSamples = 2
    numEpochs = 1

    # configure the DQN
    agent = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, seed=seedValue, verbosity=Verbosity.VERBOSE)

    _ = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)

    print(agent)

# Task1's test
def DqnLabTest3(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 3 - DQN agent on a harder Parking MDP, with Seed " + str(seedValue))

    mdp, start = createParkingMDP("basicParkingLot")

    # configure the DQN
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization
    agent = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes, regularization=regularization, seed=seedValue, verbosity=Verbosity.SILENT)

    numSamples = 50
    numEpochs = 50
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
    print(agent.analyzeQfn())

    # plot the results
    plt.title("Rewards for " + mdp.name)
    plt.plot(range(numEpochs), rewards, label=agent.name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.gca().set_ylim(bottom=-1000, top=100) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()


# Task 2's test
def DqnLabTest4(seedValue):
    '''
    probGreedy in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 4 - Comparing 3 (or more) Q learning agents, with Seed " + str(seedValue))

    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=.9)

    # configure the DQNs
    layer_sizes = [128]  # make a 1 layer MLP with 128 neurons
    regularization = .0001  # This is the default amount of regularization

    probGreedy = .999
    agent1 = DeepQNetworkAgent("BasicDQNgreedHIGH", mdp.numActions, mdp.numStates, probGreedy=probGreedy, seed=seedValue)

    probGreedy = .7
    agent2 = DeepQNetworkAgent("BasicDQNgreedMED", mdp.numActions, mdp.numStates, probGreedy=probGreedy, seed=seedValue)

    probGreedy = .001
    agent3 = DeepQNetworkAgent("BasicDQNgreedLOW", mdp.numActions, mdp.numStates, probGreedy=probGreedy, seed=seedValue)

    numSamples = 50
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the PROB GREEDY of 3 agents")
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

# Task 3's test
def DqnLabTest5(seedValue):
    '''
    Learning rate in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - Comparing Q learning agents in 3 (or more) different learning rates, with Seed " + str(seedValue))

    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate)

    learningRate = 0.01
    agent1 = DeepQNetworkAgent("BasicDQNlearningRateHIGH", mdp.numActions, mdp.numStates, learningRate=learningRate)

    learningRate = 0.0001
    agent2 = DeepQNetworkAgent("BasicDQNlearningRateMED(default)", mdp.numActions, mdp.numStates, learningRate=learningRate)

    learningRate = 0.00001
    agent3 = DeepQNetworkAgent("BasicDQNlearningRateLOW", mdp.numActions, mdp.numStates, learningRate=learningRate)

    numSamples = 50
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the LEARNING RATE of 3 agents")
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

def DqnLabTest6(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 6 - Comparing 3 (or more) MDPs, with Seed " + str(seedValue))

    busyRate = .99
    # all the mdps are the same size, so we can re-use the start index
    mdp1, start = createParkingMDP("VERYbusyParkingLot", busyRate=busyRate)

    busyRate = .5
    mdp2, _ = createParkingMDP("halfBusyParkingLot", busyRate=busyRate)

    busyRate = .2
    mdp3, _ = createParkingMDP("mostlyEmptyParkingLot", busyRate=busyRate)

    numSamples = 50
    numEpochs = 50
    mdps = [mdp1, mdp2, mdp3]
    allRewards = []
    for i in range(len(mdps)):
        agent = DeepQNetworkAgent("BasicDQN", mdps[i].numActions, mdps[i].numStates, seed=seedValue)
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdps[i], start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the BUSY RATE of 3 mdps with agent " + agent.name)
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=mdps[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=mdps[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=mdps[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

# Task 4's test, network depth
def DqnLabTest7(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 7 - Comparing DQN agents with 3 (or more) different depths, with Seed " + str(seedValue))

    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate)

    # configure the DQNs
    layer_sizes = [64, 64]  # make a 2-layer MLP with 64 neurons on each layer (128 total neurons)
    agent1 = DeepQNetworkAgent("BasicDQNlayers2", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    layer_sizes = [43, 43, 43]  # make a 3-layer MLP with 43 neurons on each layer (129 total neurons)
    agent2 = DeepQNetworkAgent("BasicDQNlayers3", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    layer_sizes = [32, 32, 32, 32]  # make a 4-layer MLP with 32 neurons on each layer (128 total neurons)
    agent3 = DeepQNetworkAgent("BasicDQNlayers4", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    numSamples = 50
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the DEPTH of 3 agents with MDP " + mdp.name)
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

# Task 5's test, network width
def DqnLabTest8(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 8 - Comparing DQN agents with 3 (or more) different widths, with Seed " + str(seedValue))

    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=busyRate)

    layer_sizes = [64]
    agent1 = DeepQNetworkAgent("BasicDQNnarrower", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    layer_sizes = [128]
    agent2 = DeepQNetworkAgent("BasicDQN", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    layer_sizes = [256]
    agent3 = DeepQNetworkAgent("BasicDQNwider", mdp.numActions, mdp.numStates, layer_sizes=layer_sizes)

    numSamples = 50
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the WIDTH of 3 agents with MDP " + mdp.name)
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

# Task 6's test, a more/less regularized network
def DqnLabTest9(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 9 - Comparing DQN agents with 3 (or more) different regularizations, with Seed " + str(seedValue))

    busyRate = .9
    mdp, start = createParkingMDP("busierParkingLot", busyRate=.9)

    # configure the DQNs
    regularization = .0001  # This is the default amount of regularization
    agent1 = DeepQNetworkAgent("BasicDQNregularizationDefault", mdp.numActions, mdp.numStates, regularization=regularization)

    regularization = 0  # Do no regularization
    agent2 = DeepQNetworkAgent("BasicDQNregularizationNONE", mdp.numActions, mdp.numStates, regularization=regularization)

    regularization = .01
    agent3 = DeepQNetworkAgent("BasicDQNregularizationHIGH", mdp.numActions, mdp.numStates, regularization=regularization)

    numSamples = 50
    numEpochs = 50
    agents = [agent1, agent2, agent3]
    allRewards = []
    for agent in agents:
        rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=50)
        allRewards.append(rewards)
        print()

    plt.title("Rewards varying the REGULARIZATION of 3 agents with MDP " + mdp.name)
    plt.plot(range(numEpochs), allRewards[0], 'peru', label=agents[0].name)
    plt.plot(range(numEpochs), allRewards[1], 'skyblue', label=agents[1].name)
    plt.plot(range(numEpochs), allRewards[2], "mediumseagreen", label=agents[2].name)
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.gca().set_ylim(bottom=0, top=200) #you may find it helpful to restrict the Y axis if you want to focus on some part of output
    plt.show()

# Assignment Task (#7)'s test, Build the best DQN you can for this random MDP
def DqnLabTest10(seedValue):
    '''
    Learning rate in the Q-learning will be varied in this test
    It will create the following files for you the generate the learning curve:
        {agentname}_Large Parking MDP_rewardCurveData.txt: Average reward of each epoch
        {agentname}_Large Parking MDP_rewardList.txt: Reward List of each epoch
    '''
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 10 - Build the best DQN you can for this random MDP, with Seed " + str(seedValue))

    numStates = 45
    numActions = 15
    start = 0
    mdp = generate_random_mdp(numStates, numActions)

    # TODO change the below configs to create the best agent you are able!!! Try not to just make the NN huge
    probGreedy = .7
    discountFactor = .9
    learningRate = .001
    layer_sizes = [128]
    regularization = .0001
    scheduleLR = False
    agent = DeepQNetworkAgent("YourDQN", numActions, mdp.numStates, probGreedy, discountFactor, learningRate,
                              layer_sizes, regularization, seedValue, Verbosity.SILENT)

    # Try to keep these parameters fixed so you arent spending huge amounts of compute on the problem
    numSamples = 50
    numEpochs = 50
    maxTrajectoryLen = 100

    # If you want to compare multiple agent variations, feel free to modify code framework from earlier tests
    rewards = trainingAndTestLoop(numEpochs, numSamples, mdp, start, agent, maxTrajectoryLen=maxTrajectoryLen)

    plt.title("Rewards for " + agent.name + " on " + mdp.name)
    plt.plot(range(numEpochs), rewards, 'peru')
    plt.ylabel('Rewards')
    plt.xlabel('Epoch')
    plt.show()