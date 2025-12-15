from enum import Enum

class Act(Enum):
    PARK = 0
    DRIVE = 1

    # this function handles print
    def __repr__(self):
        if self.value == Act.PARK.value:
            return "PARK "
        elif self.value == Act.DRIVE.value:
            return "drive "
        else:
            print("PANIC!!!! UNKNOWN TYPE in Action::repr: ", self.value)
            return ""

    # this function handles calls to str()
    def __str__(self):
        return self.__repr__()

#################################
class StateType(Enum):
    PARKED = 0
    CRASHED = 1
    DRIVING_OCCUPIED = 2
    DRIVING_AVAILABLE = 3

    # this function handles print
    def __repr__(self):
        if self.value == StateType.PARKED.value:
            return "Parked "
        elif self.value == StateType.CRASHED.value:
            return "Crashed "
        elif self.value == StateType.DRIVING_OCCUPIED.value:
            return "Driving (Occupied)"
        elif self.value == StateType.DRIVING_AVAILABLE.value:
            return "Driving (Available)"
        else:
            print("PANIC!!!! UNKNOWN TYPE in State::repr: ", self.value)
            return ""

    # this function handles calls to str()
    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get(state):
        return StateType(state % 4)

    @staticmethod
    def getIndices(state, numSpacesPerRow):
        # in python, // means integer division
        currSpaceIdx = (state // 4) % numSpacesPerRow
        currRowIdx = (state // 4) // numSpacesPerRow
        return currRowIdx, currSpaceIdx

    @staticmethod
    def lapFinished(iteration, numStates, factor):
        return iteration > (numStates / 4)*factor

    @staticmethod
    def interpretState(state, numSpacesPerRow):
        stateClass = StateType.get(state)
        currRowIdx, currSpaceIdx = StateType.getIndices(state, numSpacesPerRow)
        return "Interpreted state: [" + str(currRowIdx) + "," + str(currSpaceIdx) + "] " + str(stateClass)

