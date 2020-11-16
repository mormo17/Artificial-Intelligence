# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        


    
    def runValueIteration(self):
        # Write value iteration code here
        
        for indx in range(self.iterations):
            new_values = {}
            for curr_state in self.mdp.getStates():
                if self.mdp.isTerminal(curr_state):
                    new_values[curr_state] = 0
                else:
                    new_action = float('-inf')
                    for possible_action in self.mdp.getPossibleActions(curr_state):
                        new_action = max(new_action, self.computeQValueFromValues(curr_state, possible_action))
                    new_values[curr_state] = new_action
            
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        result = 0
        value = 0
        
        for (next_state, next_prob) in states_probs:
            value = self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]
            result += value * next_prob
        
        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None
        else:
            result_action = None
            result_value = float('-inf')
            for possible_action in self.mdp.getPossibleActions(state):
                q_val = self.computeQValueFromValues(state, possible_action)
                if result_value < q_val:
                    result_value = q_val
                    result_action = possible_action     # We want result_action to be action with maximum q_value

            return result_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.values = {}
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        
        for state in self.mdp.getStates():
            self.values[state] = 0

        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            self.update(state)

    def update(self, state):
        values = []
        if not self.mdp.isTerminal(state):
            for possible_action in self.mdp.getPossibleActions(state): 
                values.append(self.computeQValueFromValues(state, possible_action))
            self.values[state] = (max(values))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        values = {}
        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for possible_action in self.mdp.getPossibleActions(state):
                    for next_state, next_prob in self.mdp.getTransitionStatesAndProbs(state, possible_action):
                        if  next_state not in values:
                            values[next_state] = {state}
                        else:
                            values[next_state].add(state)

                max_value = max(map(lambda action: self.getQValue(state, action), self.mdp.getPossibleActions(state)))
                
                diff  =  abs(self.values[state] - (max_value))
                pq.push(state, -diff)
        self.update(pq, values)
    
    def update(self, pq, values):
        for _ in range(self.iterations):
            if not pq.isEmpty():
                state = pq.pop()
                self.values[state] = max(map(lambda action: self.getQValue(state, action), self.mdp.getPossibleActions(state)))

                for curr_value in values[state]:
                    final_max = max(map(lambda action: self.getQValue(curr_value,action), self.mdp.getPossibleActions(curr_value)))
                    diff = (abs(self.values[curr_value] - final_max))
                    if diff > self.theta:
                        pq.update(curr_value, -diff)
            else:
                break
               