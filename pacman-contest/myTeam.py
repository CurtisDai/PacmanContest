# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random,  util
from util import nearestPoint
from game import Directions
import game,operator


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveDummyAgent', second = 'DefensiveDummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.weights = util.Counter()
    self.walls = gameState.getWalls()
    self.mapArea = self.walls.width * self.walls.height
    self.q_table = util.Counter()
    self.epsilon = 0.2
    self.alpha = 0.2
    self.discount = 0.6
    self.max_episodes = 10
    self.last_action = 'Stop'
    self.smart_action = ['North', 'South', 'East', 'West']
    self.position = gameState.getAgentState(self.index).getPosition()
    self.start = gameState.getAgentPosition(self.index)
    self.last_foodlist = self.getFood(gameState).asList()
    self.count = 0
    self.fake_walls = self.walls
    global assumes
    self.mostLike = [None] * 4
    self.powerTimer = 0
    self.patrol = False
    self.last_food_defend = self.getFoodYouAreDefending(gameState).asList()
    self.x, self.y = gameState.getWalls().asList()[-1]
    self.avaliablePositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.enemy_eat_food_now = False
    self.enemy_there = None

    self.chokes = []

    self.capsule_time = 120

    if self.red:
        xAdd = -2
    else:
        xAdd = 3
    # Find all choke points of interest
    for i in range(self.y):
        if not self.walls[self.x / 2 + xAdd][i]:
            self.chokes.append(((self.x / 2 + xAdd), i))
    if self.index == max(gameState.getRedTeamIndices()) or self.index == max(gameState.getBlueTeamIndices()):
        x, y = self.chokes[3 * len(self.chokes) / 4]
    else:
        x, y = self.chokes[1 * len(self.chokes) / 4]

    self.goalTile = (x, y)


    assumes = [util.Counter()] * gameState.getNumAgents()

    # All assumes begin with the agent at its inital position
    for i, val in enumerate(assumes):
        if i in self.getOpponents(gameState):
            j = gameState.getInitialAgentPosition(i)
            assumes[i][j] = 1.0


    self.goToCenter(gameState)
    self.initial_target = self.getPossibleTarget(gameState)
    if not self.initial_target:
        self.initial_target = self.goToCenter(gameState)

    self.halflines = self.backLine()
    self.frontlines = self.frontLine()

    # self.current_state = self.getCurrentObservation()
    # self.PreQlearn(gameState,50)




    '''
    Your initialization code goes here, if you need any.
    '''




  #
  # def getQValue(self, state, action):
  #   return self.q_table[(state,action)]
  def NStepQlearn(self,state,n):

    self.q_table = util.Counter()
    self.position = state.getAgentState(self.index).getPosition()
    for episode in  range(self.max_episodes):
      step = 0
      round_state = state
      self.last_action = 'Stop'
      while step < n:
        # if util.flipCoin(self.epsilon):
        #   actions = self.getMoveActions(round_state)
        #   action = random.choice(actions)
        # else:
        actions = self.getBetterAction(self.last_action,round_state)
        action = self.computeActionFromQValues2(round_state,actions)
        # if action != 'Stop':
        # print action
        if action:
          nextstate = self.getSuccessor(round_state,action)

          reward = self.getQValue(round_state,action)

          self.update(round_state, action, nextstate, reward)

          round_state = nextstate

        # print round_state.getAgentState(self.index).getPosition()
          self.last_action = action
        if len(round_state.getLegalActions(self.index)) < 2:
            step = step + n
        step += 1


  def getFeatures(self, state, action):
      # extract the grid of food and wall locations and get the ghost locations
      food = self.getFood(state)
      see_ghost,ghosts_pos,min_distance =self.seeGhost(state)
      next_state = self.getSuccessor(state, action)
      # print ghosts
      features = util.Counter()
      features["bias"] = 1.0

      # compute the location of pacman after he takes the action

      next_pos = next_state.getAgentState(self.index).getPosition()
      next_x,next_y = next_pos

      # count the number of ghosts 1-step away

      if see_ghost and not self.isScared(next_state) and next_state.getAgentState(self.index).isPacman:

          features["#-of-ghosts-1-step-away"] = - 10 * sum(next_pos in
                                                           game.Actions.getLegalNeighbors(g, self.walls) for g in ghosts_pos)


          if len(game.Actions.getLegalNeighbors(next_pos, self.walls)) < 3:
              features["dead-way"] = -100.0

          if not next_state.getAgentState(self.index).isPacman:
              features["go-back"] = 0.2

          if self.getCapsules(state) is not None:
            if (int(next_x),int(next_y)) in self.getCapsules(state):
              features["distance-to-capsules"] = 10000.0
            else:
              features["distance-to-capsules"] = 10.0/self.DistanceToCapsules(next_pos,next_state)

          dis_to_ghost = self.DistanceToGhost(next_pos, ghosts_pos)
          if dis_to_ghost  == 0 or dis_to_ghost == 1:
            features["distance-to-ghost"] = -10000.0
            # print features
          else:
            features["distance-to-ghost"] = - 10.0 / dis_to_ghost



      if not features["#-of-ghosts-1-step-away"] and food[int(next_x)][int(next_y)] and not features["dead-way"]:
          features["eats-food"] = 100.0


      dist = self.closestFood((int(next_x), int(next_y)), food, self.fake_walls)

      if dist is not None:

          if dist == 0:
            features["closest-food"] = 10.0
          else:
            features["closest-food"] =1.0 / dist
      else:
          dist = self.closestFood((int(next_x), int(next_y)), food, self.walls)
          if dist:
              if dist == 0:
                  features["closest-food"] = 10.0
              else:
                  features["closest-food"] = 1.0 / dist



      features.divideAll(10.0)
      return features

  def goToCenter(self, gameState):
      locations = []
      self.atCenter = False
      x = gameState.getWalls().width / 2
      y = gameState.getWalls().height / 2
      # 0 to x-1 and x to width
      if self.red:
          x = x - 1
      # Set where the centre is
      self.center = (x, y)
      maxHeight = gameState.getWalls().height

      # Look for locations to move to that are not walls (favor top positions)
      for i in xrange(maxHeight - y):
          if not gameState.hasWall(x, y):
              locations.append((x, y))
          y = y + 1

      myPos = gameState.getAgentState(self.index).getPosition()
      minDist = float('inf')
      minPos = None

      # Find shortest distance to centre
      for location in locations:
          dist = self.getMazeDistance(myPos, location)
          if dist <= minDist:
              minDist = dist
              minPos = location

      return minPos

  def getQValue(self, state, action):

      return 0




  def evaluate(self, gameState, action, evaluateType):
    """
    Computes a linear combination of features and feature weights
    """
    pass

  def closestFood(self, pos, food, walls):

      fringe = [(pos[0], pos[1], 0)]
      expanded = set()
      while fringe:
          pos_x, pos_y, dist = fringe.pop(0)
          if (pos_x, pos_y) in expanded:
              continue
          expanded.add((pos_x, pos_y))
          # if we find a food at this location then exit
          if food[pos_x][pos_y]:
              return dist
          # otherwise spread out from the location to its neighbours
          nbrs = game.Actions.getLegalNeighbors((pos_x, pos_y), walls)
          for nbr_x, nbr_y in nbrs:
              fringe.append((nbr_x, nbr_y, dist + 1))
      # no food found
      return None

  def closestRoad(self, pos, enemy, walls):

      fringe = [(pos[0], pos[1], 0)]
      expanded = set()
      min_distance = 999
      new_wall =walls
      for tile in self.frontlines:
          x,y = tile
          new_wall[x][y] = True

      h_x,h_y = enemy
      while fringe:
          pos_x, pos_y, dist = fringe.pop(0)
          if (pos_x, pos_y) in expanded:
              continue
          expanded.add((pos_x, pos_y))

          if int(pos_x) == int(h_x) and int(pos_y) == int(h_y):
              return dist
          nbrs = game.Actions.getLegalNeighbors((pos_x, pos_y), new_wall)
          for nbr_x, nbr_y in nbrs:
              fringe.append((nbr_x, nbr_y, dist + 1))
          # no food found
      return min_distance

  def closestBackWay(self, pos, lines, walls):
      fringe = [(pos[0], pos[1], 0)]
      expanded = set()
      min_distance = None
      for bound in lines:
          h_x, h_y = bound
          while fringe:
              pos_x, pos_y, dist = fringe.pop(0)
              if (pos_x, pos_y) in expanded:
                  continue
              expanded.add((pos_x, pos_y))
              if int(pos_x) == int(h_x) and int(pos_y) == int(h_y):
                  if min_distance is None or dist < min_distance:
                    min_distance = dist
                  break

              nbrs = game.Actions.getLegalNeighbors((pos_x, pos_y), walls)
              for nbr_x, nbr_y in nbrs:
                  fringe.append((nbr_x, nbr_y, dist + 1))

      return min_distance

  def closestFoodPos(self,pos,food,walls):
      fringe = [(pos[0], pos[1],0)]
      expanded = set()
      while fringe:
          pos_x, pos_y,dist = fringe.pop(0)
          if (pos_x, pos_y) in expanded:
              continue
          expanded.add((pos_x, pos_y))
          if food[pos_x][pos_y]:
              return (pos_x,pos_y)
          nbrs = game.Actions.getLegalNeighbors((pos_x, pos_y), walls)
          for nbr_x, nbr_y in nbrs:
              fringe.append((nbr_x, nbr_y, dist + 1))
      # no food found
      return None

  def computeValueFromQValues(self, state):
    q_value_set = []
    for action in state.getLegalActions(self.index):
      q_value_set.append(self.getQValue(state, action))
    if len(state.getLegalActions(self.index)) == 0:
      return 0.0
    else:
      return max(q_value_set)

  def getPossibleTarget(self, gameState):
      pos = gameState.getAgentState(self.index).getPosition()
      food = self.getFood(gameState)
      x, y = pos
      x = int(x)
      y = int(y)
      food_pos = self.closestFoodPos((x, y), food, self.walls)

      x, y = food_pos
      max_x = self.walls.width - 1
      max_y = self.walls.height - 1
      target = None
      if food_pos and not gameState.hasWall(max_x - x, max_y - y):
          target = (max_x - x, max_y - y)
      return target
  def getActionFromQValues(self,state):
      max_q = -999
      max_action = 'Stop'
      for action in self.getMoveActions(state):
          q_val = self.q_table[(state, action)]
          if q_val > max_q:
              max_action = action
              max_q = q_val
      return max_action


  def getMoveActions(self, state):
      actions = state.getLegalActions(self.index)
      if 'Stop' in actions:
          actions.remove('Stop')
      return actions

  def getOppositeAction(self,action):
      direction = {
          'West': 'East',
          'East': 'West',
          'South': 'North',
          'North': 'South',
          'Stop': 'Stop',
      }
      return direction.get(action,'Stop')

  def getBetterAction(self,action,state):
      actions = self.getMoveActions(state)

      oppsoite_action = self.getOppositeAction(action)
      if oppsoite_action in actions:
        actions.remove(oppsoite_action)

      return actions

  def backLine(self):
      x = self.walls.width/2
      back_tiles = []
      if self.red:
          x = x-1
      for y in range(self.walls.height):
          if not self.walls[x][y]:
              back_tiles.append((x,y))
      return back_tiles

  def frontLine(self):
      x = self.walls.width/2
      front_tiles = []
      if not self.red:
          x = x-1
      for y in range(self.walls.height):
          if not self.walls[x][y]:
              front_tiles.append((x,y))
      return front_tiles


  def update(self, state, action, nextState, reward):

    first_part = (1 - self.alpha) * self.getQValue(state, action)
    actions = self.getBetterAction(action,nextState)
    if len(actions) == 0:
      sample = reward
    else:
      sample = reward + (self.discount * max(
        [self.getQValue(nextState, next_action) for next_action in actions]))
    second_part = self.alpha * sample
    self.q_table[(state, action)] = first_part + second_part


  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def getOpponentsPos(self, gameState):

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    return ghosts

  def getOpponentsPacmanPos(self,gameState):
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      pacman = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
      return pacman

  def DistanceToCapsules(self, myPos, state):
      positions = self.getCapsules(state)
      distances = []
      for pos in positions:
          distances.append(self.getMazeDistance(myPos, pos))
      if len(distances) > 0:
          return min(distances)
      else:
          return 999

  def run(self,state,ghosts):
      bestDist = 0
      bestAction = 'Stop'
      actions = self.getMoveActions(state)

      for action in actions:
          feature = util.Counter()
          successor = self.getSuccessor(state, action)
          pos2 = successor.getAgentPosition(self.index)
          see_ghost, ghosts_pos, min_distance = self.seeGhost(successor)

          dist = self.closestBackWay(pos2,self.halflines, self.fake_walls)
          if dist:
              feature['back'] =10.0/(dist+1)
          else:
              self.fake_walls = self.walls
              dist = self.closestBackWay(pos2, self.halflines, self.walls)
              if dist:
                  feature['back'] = 10.0 / (dist+1)


          if not successor.getAgentState(self.index).isPacman:
              feature['home'] =  1000
          if see_ghost and not self.isScared(state):
              danger = min_distance
              if danger < 2:
                  feature['dead'] = -10000000
              else:
                  feature['dead'] = - 1.0/danger
              nbr = game.Actions.getLegalNeighbors(pos2, self.walls)
              if len(nbr) == 1:
                  feature['dead-way'] = -1000
              distance = self.DistanceToCapsules(pos2, state)
              if distance < 1:
                  feature['power'] =  1000000
              else:
                  feature['power'] = 100.0 / distance
          else:
              food = self.getFood(state)
              x,y = pos2
              if food[int(x)][int(y)]:
                  feature['eat'] =  10
          if feature.totalCount()> bestDist:
              bestAction = action
              bestDist = feature.totalCount()
      return bestAction

  def isdead(self,state):
      if self.getPreviousObservation():
          if self.getPreviousObservation().getAgentState(self.index).isPacman:
              if not state.getAgentState(self.index).isPacman:
                  return True

  def pacmanCome(self,gameState):
      opponent = self.getOpponents(gameState)
      for index in opponent:
        if gameState.getAgentState(index).isPacman:
            return True

  def isScared(self,state):
      indices = self.getOpponents(state)
      for index in indices:
          if state.data.agentStates[index].scaredTimer > 0:
              return True
      return False

  def DistanceToGhost(self,myPos,ghostsPos):
      distance = []
      for g in ghostsPos:
          distance.append(self.getMazeDistance(myPos, g))
      return min(distance)



  def computeActionFromQValues(self, state):
    # max_q = self.computeValueFromQValues(state)
    max_q = -10000000
    max_action = 'Stop'
    for action in self.getMoveActions(state):
      q_val = self.getQValue(state, action)
      if q_val > max_q:
          max_action = action
          max_q = q_val
    return max_action

  def computeActionFromQValues2(self, state,actions):
    # max_q = self.computeValueFromQValues(state)
    max_q = -100000000
    max_action = None
    for action in actions:
      q_val = self.getQValue(state, action)
      if q_val > max_q:
          max_action = action
          max_q = q_val
    return max_action

  def seeGhost(self,gameState):
      ghosts = self.getOpponentsPos(gameState)
      myPos = gameState.getAgentPosition(self.index)
      ghosts_pos= []
      min_distance = 10
      if ghosts:
          for g in ghosts:
              distance = self.getMazeDistance(myPos, g)
              if distance < min_distance:
                  min_distance = distance
              if distance <6:
                  ghosts_pos.append(g)
          if min_distance <6:
            return True,ghosts_pos,min_distance
      return False,None,999


class OffensiveDummyAgent(DummyAgent):
  """
  A reinforcement agent use greedy method to train Q-table and use Qlearning to find food"
  """

  def chooseAction(self, gameState):
    see_ghost,ghosts_pos,min_distance = self.seeGhost(gameState)
    if self.isScared(gameState) or self.isdead(gameState):
        self.fake_walls = gameState.getWalls()
    # ghosts = self.getOpponentsPos(gameState)

    foodlist = self.getFood(gameState).asList()

    if see_ghost and not self.isScared(gameState):
        self.fake_walls = gameState.getWalls()
        for (x,y) in ghosts_pos:
            for (wall_x,wall_y) in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
              if wall_x <= self.walls.width and wall_x >= 0 and wall_y <=self.walls.height and wall_y >= 0:
                self.fake_walls[int(wall_x)][int(wall_y)] = True
    if see_ghost and gameState.getAgentState(self.index).isPacman and not self.isScared(gameState) and min_distance<3:
        action = self.run(gameState,ghosts_pos)
    else:
        if len(self.last_foodlist) - len(foodlist) > 4:
          action = self.run(gameState, ghosts_pos)
        else:
          self.NStepQlearn(gameState, 10)
          action = self.getActionFromQValues(gameState)


    self.last_action = action
    if action != 'Stop':
      if not self.getSuccessor(gameState, action).getAgentState(self.index).isPacman:
        self.last_foodlist = self.getFood(gameState).asList()

    return action




  def chaseMe(self,myPos,ghostsPos):
      for g in ghostsPos:
          if myPos in game.Actions.getLegalNeighbors(g, self.walls):
              return True
      return False

  def getQValue(self, state, action):
      if self.q_table[(state,action)] == 0:

          features = self.getFeatures(state, action)

          q_value = features.totalCount()
          self.q_table[(state,action)] = q_value
      return self.q_table[(state,action)]

  def update(self, state, action, nextState, reward):

    first_part = (1 - self.alpha) * self.q_table[(state, action)]
    actions = self.getBetterAction(action,nextState)
    if len(actions) == 0:
        sample = reward
    else:
        sample = reward + (self.discount * max(
            [self.getQValue(nextState, next_action) for next_action in actions]))
    second_part = self.alpha * sample
    self.q_table[(state, action)] = first_part + second_part

class DefensiveDummyAgent(DummyAgent):
    # Detect position of enemies
    def getEnemyPosition(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemyPos.append((enemy, pos))
        return enemyPos

    # Find which enemy is the closest
    def enemyClose(self, gameState):
        pos = self.getEnemyPosition(gameState)
        closest = None
        if len(pos) > 0:
            closest = float('inf')
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < closest:
                    closest = dist
        return closest


    def getDistToMate(self, gameState):
        indices = self.getTeam(gameState)
        distance = 999
        if len(indices) ==2:
            distance = self.getMazeDistance(gameState.getAgentPosition(indices[0]),
                                            gameState.getAgentPosition(indices[1]))
        return distance


    # Is pacman powered?
    def isPowered(self):
        return self.powerTimer > 0

    # How much longer is the ghost scared?
    def ScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    # Gets the distribution for where a ghost could be, all weight equally
    def getDistribution(self, p):
        posActions = [(p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1), (p[0], p[1])]
        actions = []
        for act in posActions:
            if act in self.avaliablePositions:
                actions.append(act)

        distribution = util.Counter()
        for act in actions:
            distribution[act] = 1
        return distribution



    def elapseTime(self, gameState):
        for agent, assume in enumerate(assumes):
            if agent in self.getOpponents(gameState):
                newBeliefs = util.Counter()
                # Checks to see what we can actually see
                pos = gameState.getAgentPosition(agent)
                if pos != None:
                    newBeliefs[pos] = 1.0
                else:
                    # Look at all current beliefs
                    for p in assume:
                        if p in self.avaliablePositions and assume[p] > 0:
                            # Check that all these values are legal positions
                            newPosDist = self.getDistribution(p)
                            for x, y in newPosDist:  # iterate over these probabilities
                                newBeliefs[x, y] += assume[p] * newPosDist[x, y]
                                # The new chance is old chance * prob of this location from p
                    if len(newBeliefs) == 0:
                        oldState = self.getPreviousObservation()
                        if oldState != None and oldState.getAgentPosition(agent) != None:  # just ate an enemy
                            newBeliefs[oldState.getInitialAgentPosition(agent)] = 1.0
                        else:
                            for p in self.avaliablePositions: newBeliefs[p] = 1.0
                assumes[agent] = newBeliefs

        # self.displayDistributionsOverPositions(assumes)

    # Search where the enemies currently are
    def observe(self, agent, noisyDistance, gameState):
        myPos = gameState.getAgentPosition(self.index)
        # Current state probabilities
        allPossible = util.Counter()
        for p in self.avaliablePositions:  # check each legal position
            trueDistance = util.manhattanDistance(p, myPos)  # distance between this point and Pacman
            allPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)
            # The new values are product of prior probability and new probability
        for p in self.avaliablePositions:
            assumes[agent][p] *= allPossible[p]


            # Choose which action will result in the best move

    def chooseAction(self, gameState):

        opponents = self.getOpponents(gameState)
        # Get noisey distance data
        noisyData = gameState.getAgentDistances()


        for agent in opponents:
            self.observe(agent, noisyData[agent], gameState)


            # Set default move location to the hover position from above
        self.locations = [self.chokes[len(self.chokes) / 2]] * gameState.getNumAgents()
        for i, assume in enumerate(assumes):
            maxLoc = 0
            # max location
            checkForAllEq = 0
            for val in assumes[i]:
                # Checks if there are many possible locations for the enemy with equal probability
                if assume[val] == maxLoc and maxLoc > 0:
                    # If many locations are equally likely, ignore this inference itteration as it is inaccurate
                    checkForAllEq += 1
                elif assume[val] > maxLoc:
                    maxLoc = assume[val]
                    self.locations[i] = val
                    # Set target location as the highest probability location
            if checkForAllEq > 5:
                self.locations[i] = self.goalTile

        # Normalise new probabilities and pick most likely location for enemy agent
        for agent in opponents:
            assumes[agent].normalize()
            self.mostLike[agent] = max(assumes[agent].iteritems(), key=operator.itemgetter(1))[0]
        self.elapseTime(gameState)
        # Get agent position
        agentPos = gameState.getAgentPosition(self.index)

        food_defend = self.getFoodYouAreDefending(gameState).asList()
        if len(food_defend) < len(self.last_food_defend):
            for food in self.last_food_defend:
                if food not in food_defend:
                    self.enemy_eat_food_now = True
                    self.enemy_there = food

        if self.enemy_eat_food_now:
            if self.getMazeDistance(agentPos,self.enemy_there) < 5:
                self.enemy_eat_food_now = False
                self.enemy_there = None

        if not self.patrol:
            evaluateType = 'start'
        else:
            evaluateType = 'patrol'

        if self.last_action == 'Stop':
            self.count += 1
            if self.count > 2:
                self.count = 0
                self.patrol = True



        # If an enemy is attacking our food, hunt that enemy down
        for agent in opponents:
            if (gameState.getAgentState(agent).isPacman):
                evaluateType = 'hunt'

                # If we directly see an enemy on our side, swich to defence
        enemyPos = self.getEnemyPosition(gameState)

        if len(enemyPos) > 0:
            closest_pos = None
            minDistance = 100
            for enemy, pos in enemyPos:
                if self.getMazeDistance(agentPos, pos) < minDistance:
                    closest_pos = pos
                    minDistance = self.getMazeDistance(agentPos, pos)
            if minDistance < 5 and not gameState.getAgentState(self.index).isPacman:
                    evaluateType = 'defend'




        # Get all legal actions this agent can make in this state
        actions = gameState.getLegalActions(self.index)
        # Calcualte heuristic score of each action
        heuristicVal = [self.evaluate(gameState, a, evaluateType) for a in actions]
        # Pick the action with the highest heuristic score as the best next move
        maxValue = max(heuristicVal)
        bestActions = [a for a, v in zip(actions, heuristicVal) if v == maxValue]

        # If multiple best moves exist (unlikely), pick one at random
        action = random.choice(bestActions)
        self.last_action = action
        self.last_food_defend = self.getFoodYouAreDefending(gameState).asList()
        return action



    # Calculate the heurisic score of each action depending on what tactic is being used
    def evaluate(self, gameState, action, evaluateType):

        if evaluateType == 'start':

            features = self.getFeaturesStart(gameState, action)
            weights = self.getWeightsStart(gameState, action)
        elif evaluateType == 'patrol':

            features = self.getFeaturesPatrol(gameState, action)
            weights = self.getWeightsPatrol(gameState, action)
        elif evaluateType == 'hunt':

            features = self.getFeaturesHunt(gameState, action)
            weights = self.getWeightHunt(gameState, action)
        elif evaluateType == 'defend':

            features = self.getFeaturesDefend(gameState, action)
            weights = self.getWeightsDefend(gameState, action)
        else :
            return 0
        return features * weights


    # Returns all the heuristic features for the START tactic
    def getFeaturesStart(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Get own position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Compute distance to board centre
        dist = self.getMazeDistance(myPos, self.initial_target)
        features['distToCenter'] = dist
        if myPos == self.center:
            features['atCenter'] = 1
        return features

    # Returns all the heuristic features for the HUNT tactic
    def getFeaturesPatrol(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        opponents = self.getOpponents(successor)
        enemyDist = []
        for agent in opponents:
            enemyPos = self.mostLike[agent]
            enemyDist.append(self.getMazeDistance(myPos, enemyPos))
        features['invaderDistance'] = min(enemyDist)
        distanceToAlly = self.getDistToMate(successor)
        features['distanceToAlly'] = distanceToAlly
        if successor.getAgentState(self.index).isPacman:
            features['wrong-way'] = 1
        return features
    def getWeightsPatrol(self, gameState, action):

        return {'invaderDistance': -10, 'distanceToAlly': 1,'wrong-way': -100 }

    def getFeaturesHunt(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Get opponents and invaders
        opponents = self.getOpponents(gameState)
        invaders = [agent for agent in opponents if successor.getAgentState(agent).isPacman]

        # Find number of invaders
        features['numInvaders'] = len(invaders)

        # For each invader, calulate its most likely poisiton and distance
        food_defend = self.getFoodYouAreDefending(successor).asList()


        enemyDist = []
        for agent in invaders:
            enemyPos = self.mostLike[agent]
            enemyDist.append(self.closestRoad(myPos,enemyPos,self.walls))

        if enemyDist:
          features['invaderDistance'] = min(enemyDist)

        if self.enemy_eat_food_now:
            right_distance = self.closestRoad(myPos, self.enemy_there,self.walls)
            features['invaderDistance'] = right_distance



        # Compute distance to partner
        if successor.getAgentState(self.index).isPacman:
            distanceToAlly = self.getDistToMate(successor)
            # distanceToAgent is always None for one of the agents (so they don't get stuck)
            if distanceToAlly != None:
                features['distanceToAlly'] = 1.0 / (distanceToAlly +1)

        if action == Directions.STOP: features['stop'] = 1


        return features

    # Returns all the heuristic features for the DEFEND tactic
    def getFeaturesDefend(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Get own position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # List invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]


        # Get number of invaders
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            enemyDist = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in invaders]
            # Find closest invader
            features['invaderDistance'] = min(enemyDist)

        # Compute distance to enemy
        distEnemy = self.enemyClose(successor)
        if (distEnemy <= 5):
            features['danger'] = 1
            if (distEnemy <= 1 and self.ScaredTimer(successor) > 0):
                features['danger'] = -1
        else:
            features['danger'] = 0

        if successor.getAgentState(self.index).isPacman:
            features['wrongWay'] = 1
            distanceToAlly = self.getDistToMate(successor)
            # distanceToAgent is always None for one of the agents (so they don't get stuck)
            features['distanceToAlly'] = 1.0 / (distanceToAlly+1)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features


    # Returns heuristic weightings for the START tactic
    def getWeightsStart(self, gameState, action):
        return {'distToCenter': -1, 'atCenter': 1000}

    # Returns heuristic weightings for the HUNT tactic
    def getWeightHunt(self, gameState, action):

        return {'numInvaders': -100, 'invaderDistance': -10, 'stop': -5000,
                 'distanceToAlly': -2500}

        # Returns heuristic weightings for the DEFEND tactic

    def getWeightsDefend(self, gameState, action):
        return {'numInvaders': -10000, 'invaderDistance': -500, 'stop': -5000,
                'reverse': -200, 'danger': 3000, 'distanceToAlly': -4000,'wrongWay':-10000}










