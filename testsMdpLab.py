import matplotlib.pyplot as plt
from random import seed
from agent.AgentBase import Verbosity
from agent.RandomAgent import RandomAgent

import MDP

def mdpLabTest1():
    print("----------------------- BEGIN Test 1 - Load and Print MDP1")
    mdp = MDP.MDP(None, None, "MDP1", "data/MDP1.txt")
    print(mdp)

def mdpLabTest2():
    print("\n\n\n----------------------- BEGIN Test 2 - Load and Print MDP2")
    mdp = MDP.MDP(None, None, "MDP2", "data/MDP2.txt")
    print(mdp)

def mdpLabTest3(seedValue):
    print("\n\n\n----------------------- BEGIN Test 3 - Single trajectory on MDP1 with a random agent, with Seed " + str(seedValue))
    seed(seedValue)
    mdp = MDP.MDP(None, None, "MDP1", "data/MDP1.txt")
    agent = RandomAgent(numActions=2, verbosity=Verbosity.VERBOSE)
    _ = mdp.simulateTrajectory(agent, startState=1)

def mdpLabTest4(seedValue):
    print("\n\n\n----------------------- BEGIN Test 4 - Single trajectory on MDP2 with a random agent, with Seed " + str(seedValue))
    seed(seedValue)
    mdp = MDP.MDP(None, None, "MDP2", "data/MDP2.txt")
    agent = RandomAgent(numActions=2, verbosity=Verbosity.VERBOSE)
    _ = mdp.simulateTrajectory(agent, startState=1)

def mdpLabTest5(seedValue):
    print("\n\n\n----------------------- BEGIN Test 5 - Single trajectory on MDP3 with a random agent, with Seed " + str(seedValue))
    seed(seedValue)
    mdp = MDP.MDP(None, None, "MDP2", "data/MDP3.txt")
    agent = RandomAgent(numActions=2, verbosity=Verbosity.VERBOSE)
    _ = mdp.simulateTrajectory(agent, startState=1)

def mdpLabTest6(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 6 - Averaging results from a random agent over multiple trajectories, with Seed " + str(seedValue))

    # actually run the trajectories on the MDP
    mdp = MDP.MDP(None, None, "MDP1", "data/MDP1.txt")
    agent = RandomAgent(numActions=2, verbosity=Verbosity.RESULTS)
    numTrajectories = 10
    rewards = []
    for _ in range(numTrajectories):
        reward = mdp.simulateTrajectory(agent, startState=1)
        rewards.append(reward)
    print("\n Over " + str(numTrajectories) + " Trajectories, rewards were " + str(
        rewards) + "\n ...for a Grand Total Reward: " + str(sum(rewards)) + "\n... and average of " + str(
        sum(rewards) / numTrajectories))

    # plot the results
    plt.title("Rewards for each run of RandomAgent on MDP1")
    plt.plot(rewards, 'ro')
    plt.ylabel('Rewards')
    plt.xlabel('Run #')
    plt.show()

def mdpLabTest7(seedValue):
    seed(seedValue)
    print("\n\n\n----------------------- BEGIN Test 5 - How hard are all these MDPs?, with Seed " + str(seedValue))

    # actually run the trajectories on the MDP
    mdp1 = MDP.MDP(None, None, "MDP1", "data/MDP1.txt")
    mdp2 = MDP.MDP(None, None, "MDP2", "data/MDP2.txt")
    mdp3 = MDP.MDP(None, None, "MDP3", "data/MDP3.txt")
    mdps = [mdp1, mdp2, mdp3]
    agent = RandomAgent(numActions=2, verbosity=Verbosity.SILENT)
    numTrajectories = 10
    allRewards = []
    for i in range(len(mdps)):
        rewards = []
        print("************************* MDP ", i + 1)
        for _ in range(numTrajectories):
            reward = mdps[i].simulateTrajectory(agent, startState=1)
            rewards.append(reward)
        print("Over " + str(numTrajectories) + " Trajectories, rewards were " + str(
            rewards) + "\n ...for a Grand Total Reward: " + str(sum(rewards)) + "\n... and average of " + str(
            sum(rewards) / numTrajectories) + "\n")
        allRewards.append(rewards)

    # plot the results
    plt.title("Rewards for each run of RandomAgent on Three MDPs")
    plt.plot(range(numTrajectories), allRewards[0], 'ro', label="MDP1")
    plt.plot(range(numTrajectories), allRewards[1], 'bx', label="MDP2")
    plt.plot(range(numTrajectories), allRewards[2], "g^", label="MDP3")
    plt.ylabel('Rewards')
    plt.xlabel('Run #')
    plt.legend()
    plt.show()

