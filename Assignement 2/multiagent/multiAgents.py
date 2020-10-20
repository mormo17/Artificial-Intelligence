# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # First we need to check, if we bump into ghost
        # because we will lose; so this is worst possible scenario
        for ghostPos in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghostPos) < 2:
                return float('-inf')
        
        # Now that we checked for nearby enemies, we can
        # think about scoring more
        difference = successorGameState.getScore() - currentGameState.getScore()
        if difference > 0: 
            return 1.0 + difference

        # Not in this case (where we definetely know that there are no ghost nearby,
        # but also in all next moves we are not in benefitial position) we try to minimize
        # distance to next food
        
        minDistanceToFood = float('+inf') # should be INT_MAX initially

        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            minDistanceToFood = min(minDistanceToFood, distance)
        return 1.0 / minDistanceToFood # the closer the better

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        _, action = self.minimax(gameState)
        return action

    def minimalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        
        return Helper.minimalValue(self, gameState, depth, currentIndex)

    def maximalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        
        return Helper.maximalValue(self, gameState, depth, currentIndex)

    def minimax(self, gameState, isExpectiMax=False, isAlphaBeta=False, depth=0, currentIndex=0, alpha=float('-inf'), beta=float('inf')):
        
        return Helper.minimax(self, gameState, False, False, depth, currentIndex)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, action = self.minimax(gameState)
        return action

    def minimalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        
        return Helper.minimalValue(self, gameState, depth, currentIndex, isExpectiMax, isAlphaBeta, alpha, beta)

    def maximalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        
        return Helper.maximalValue(self, gameState, depth, currentIndex, isExpectiMax, isAlphaBeta, alpha, beta)

    def minimax(self, gameState, isExpectiMax=False, isAlphaBeta=False, depth=0, currentIndex=0, alpha=float('-inf'), beta=float('inf')):
        
        return Helper.minimax(self, gameState, False, True, depth, currentIndex, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, action = self.expectimax(gameState)
        return action


    def maximalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        return Helper.maximalValue(self, gameState, depth, currentIndex, True)


    def expValue(self, gameState, depth, currentIndex):
        result = 0
        action = None
        for legalAction in gameState.getLegalActions(currentIndex):
            nextState = gameState.generateSuccessor(currentIndex, legalAction)
            prob = 1.0 / len(gameState.getLegalActions(currentIndex))
            exp, action = self.expectimax(nextState, depth, currentIndex + 1)
            result += prob * exp
        return result, action

    def expectimax(self, gameState, depth=0, currentIndex=0):
        return Helper.minimax(self, gameState, True, False, depth, currentIndex)
            

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    """
    currentScore = scoreEvaluationFunction(currentGameState)
    currentPosition = currentGameState.getPacmanPosition()
    if currentGameState.isWin() or currentGameState.isLose():
        return currentScore

    ghostPenalty, ghostBonus = getGhostDistances(currentGameState, currentPosition)
    minFoodDist, maxFoodDist = getFoodDistances(currentGameState, currentPosition)

    if len(currentGameState.getCapsules()) < 2:
        currentScore += 150
    
    return currentScore - minFoodDist - maxFoodDist - ghostPenalty + ghostBonus

class Helper:

    def minimalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        result = float('+inf')  # should be INT_MAX initially
        for legalAction in gameState.getLegalActions(currentIndex):
            nextState = gameState.generateSuccessor(currentIndex, legalAction)
            nextValue, nextAction = self.minimax(nextState, isExpectiMax, isAlphaBeta, depth, currentIndex + 1, alpha, beta)
            result = min(result, nextValue)
            if isAlphaBeta:
                if result < alpha:
                    return result, None
                beta = min(beta, result)
        return result, None


    def maximalValue(self, gameState, depth, currentIndex, isExpectiMax=False, isAlphaBeta=False, alpha=0, beta=0):
        result = float('-inf')  # should be INT_MIN initially
        action = None
        for legalAction in gameState.getLegalActions(currentIndex):
            nextState = gameState.generateSuccessor(currentIndex, legalAction)
            if isExpectiMax:
                nextValue, nextAction = self.expectimax(nextState, depth, currentIndex + 1)
            else:
                nextValue, nextAction = self.minimax(nextState, isExpectiMax, isAlphaBeta, depth, currentIndex + 1, alpha, beta)
            result = max(result, nextValue)
            if depth == 1 and result == nextValue: 
                action = legalAction
            if isAlphaBeta:
                if result > beta:
                    return result, action
                alpha = max(alpha, result)
        return result, action
    
    def minimax(self, gameState, isExpectiMax=False, isAlphaBeta=False, depth=0, currentIndex=0, alpha=float('-inf'), beta=float('inf')):
        currentIndex = currentIndex % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if currentIndex == 0:
            if depth < self.depth:
                return self.maximalValue(gameState, depth+1, currentIndex, isExpectiMax, isAlphaBeta, alpha, beta)
            else:
                return self.evaluationFunction(gameState), None
        else:
            if isExpectiMax:
                return self.expValue(gameState, depth, currentIndex)
            return self.minimalValue(gameState, depth, currentIndex, isExpectiMax, isAlphaBeta, alpha, beta)


def getFoodDistances(currentGameState, currentPosition):
    resList = []
    foods = currentGameState.getFood()
    for food in list(foods.asList()):
        resList.append(util.manhattanDistance(food, currentPosition))
    if len(resList) == 0:
        return -1, -1
    return min(resList), max(resList)

def getGhostDistances(currentGameState, currentPosition):
    list = []
    for index in range(1, currentGameState.getNumAgents()):
        list.append(util.manhattanDistance(currentGameState.getGhostPosition(index), currentPosition))
    if len(list) == 0:
        return -1, -1
    return min(list), max(list)

# Abbreviation
better = betterEvaluationFunction
