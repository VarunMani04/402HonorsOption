from MDP import MDP

from ParkingDefs import Act

def createParkingMDP(name, numRows=2, numRegularSpacesPerRow=10, numHandicappedSpacesPerRow=5,
                     busyRate=.3, handicapBusyRate=.05,
                     parkedReward=1000, crashPenalty=-10000, waitingPenalty=-1,
                     decayBusyRate=True, decayReward=True):
    # Each space in the lot uses 4 states (Parking/Occupied are both boolean and can take 2 values)
    totalSpacesPerRow = numRegularSpacesPerRow + numHandicappedSpacesPerRow
    # compute where the agent will start (each space uses 4 states, the last state is our special terminal state, so -2
    start = 4 * totalSpacesPerRow - 2

    # We add 1 because we need an exit state to keep rewards from accruing after we are parked.
    numStates = totalSpacesPerRow * numRows * 4 + 1
    # Agent can  PARK or DRIVE.
    numActions = 2

    # setup the blank MDP, to be filled in later
    mdp = MDP(numStates, numActions, name)

    # setup the exit state
    exitState = numStates - 1
    mdp.rewardFn[exitState] = 0
    # remember the transition function is indexed ACTION - FROM - TO
    mdp.transitionFn[Act.PARK.value][exitState][exitState] = 1
    mdp.transitionFn[Act.DRIVE.value][exitState][exitState] = 1

    # Modify the transition and reward functions to accomodate each parking space
    for row in range(numRows):
        for space in range(totalSpacesPerRow):
            # compute the indices for 4 states associated with the current parking space
            parkedState = (space + totalSpacesPerRow * row) * 4
            crashedState = parkedState + 1
            drivingOccupiedState = parkedState + 2
            drivingAvailableState = parkedState + 3

            # do some fancy indexing to figure out which grid cell is next
            # this is necessary since the directionality of neighbors varies per row
            offset = 4 * (2 * (row % 2) - 1)
            nextdrivingOccupiedState = (drivingOccupiedState + offset)
            nextdrivingAvailableState = (drivingAvailableState + offset)
            if nextdrivingOccupiedState < 0:
                nextdrivingOccupiedState += 4 * (totalSpacesPerRow + 1)
            elif nextdrivingOccupiedState > numStates - 1:
                nextdrivingOccupiedState -= 4 * (totalSpacesPerRow + 1)
            if nextdrivingAvailableState < 0:
                nextdrivingAvailableState += 4 * (totalSpacesPerRow + 1)
            elif nextdrivingAvailableState > numStates - 1:
                nextdrivingAvailableState -= 4 * (totalSpacesPerRow + 1)

            # setup action 0 (Park)
            mdp.transitionFn[Act.PARK.value]        [parkedState]            [exitState]      = 1
            mdp.transitionFn[Act.PARK.value]        [crashedState]           [exitState]      = 1
            mdp.transitionFn[Act.PARK.value]        [drivingOccupiedState]   [crashedState]   = 1
            mdp.transitionFn[Act.PARK.value]        [drivingAvailableState]  [parkedState]    = 1

            # setup action 1 (Drive)
            # compute the expectation that a vehicle will occupy this parking space
            rateHere = busyRate
            if decayBusyRate:
                rateHere /= float(space + 1)
            if space < numHandicappedSpacesPerRow:
                rateHere = handicapBusyRate
            mdp.transitionFn[Act.DRIVE.value]       [parkedState]            [exitState]                  = 1
            mdp.transitionFn[Act.DRIVE.value]       [crashedState]           [exitState]                  = 1
            mdp.transitionFn[Act.DRIVE.value]       [drivingOccupiedState]   [nextdrivingOccupiedState]   = rateHere
            mdp.transitionFn[Act.DRIVE.value]       [drivingAvailableState]  [nextdrivingOccupiedState]   = rateHere
            mdp.transitionFn[Act.DRIVE.value]       [drivingOccupiedState]   [nextdrivingAvailableState]  = 1 - rateHere
            mdp.transitionFn[Act.DRIVE.value]       [drivingAvailableState]  [nextdrivingAvailableState]  = 1 - rateHere

            #setup reward function and terminal array
            mdp.rewardFn[parkedState] = parkedReward
            mdp.isTerminal[parkedState] = True
            # if appropriate, make the agent prefer parking spaces close to the store
            if decayReward:
                mdp.rewardFn[parkedState] /= float(space + 1)
            # handle rewards associated with handicapped spaces
            if (space < numHandicappedSpacesPerRow):
                mdp.rewardFn[parkedState] = -parkedReward
            mdp.rewardFn[crashedState] = crashPenalty
            mdp.isTerminal[crashedState] = True
            mdp.rewardFn[drivingOccupiedState] = waitingPenalty
            mdp.rewardFn[drivingAvailableState] = waitingPenalty
    return mdp, start