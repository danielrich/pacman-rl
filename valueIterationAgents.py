# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util, copy

from learningAgents import ValueEstimationAgent

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    
    for i in range(iterations):
        lastValues = copy.deepcopy(self.values)
        for s in mdp.getStates():
            aCounter = util.Counter()
            for a in mdp.getPossibleActions(s):
                for s2 in mdp.getStates():
                    aCounter[a] += self.T(s,a,s2) * (mdp.getReward(s,a,s2) + discount*lastValues[s2])
            self.values[s] = aCounter[aCounter.argMax()]
        
     
  def reachableStates(self, state):
      states = []
      for a in self.mdp.getPossibleActions(state):
          for (s,p) in self.mdp.getTransitionStatesAndProbs(state, a):
              states.append(s)
              
      return states

    
  def T(self, state1, action, state2):
      for (s,p) in self.mdp.getTransitionStatesAndProbs(state1, action):
          if(s == state2):
              return p
      return 0
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    sum = 0
    for state2 in self.mdp.getStates():
        sum += self.T(state,action,state2) * (self.mdp.getReward(state,action,state2) + self.discount*self.values[state2])
        
    return sum


  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    vals = util.Counter() # Vk values of states reachable from state
    for action in self.mdp.getPossibleActions(state):
        vals[action] = self.getQValue(state,action)
        
    if vals == []: return None
    return vals.argMax()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
