# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.01      # if noise is really close to zero, agent will not end up in unintended successor state
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.01       #   In this case, very small discount diminishes exit cost (which is 10)
    answerNoise = 0             #   noise equal to zero means agent will choose shortest path to reward
    answerLivingReward = -0.01  #   because reward is negative
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.01       #   In this case, very small discount diminishes exit cost (which is 10)
    answerNoise = 0.01          #   noise extremely close to zero will avoid going near risky cliff
    answerLivingReward = -0.1   #   because reward is negative
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 1          #   to choose exit rewars in ANY cases
    answerNoise = 0             #   noise equal to zero means agent will choose shortest path to reward
    answerLivingReward = -0.01  #   because of discount it should be negative
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 1          #   to choose exit rewars in ANY cases
    answerNoise = 0.01          #   noise extremely close to zero will avoid going near risky cliff
    answerLivingReward = -0.01  #   because of discount it should be negative
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 1          #   to choose exit rewars in ANY cases
    answerNoise = 0             #   noise equal to zero means agent will choose shortest path to reward
    answerLivingReward = 1      #   to stay alive forever and get infinite living points
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = 0.02
    answerLearningRate = 100
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
