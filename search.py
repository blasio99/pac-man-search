# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
class CustomNode:
     def __init__(self,name,cost):
         self.name=name
         self.cost=cost
     def getName(self):
         return self.name
     def getCost(self):
         return self.cost

class Node:
    def __init__(self,state,parent,action,path_cost):
        self.state=state
        self.parent=parent
        self.action=action
        self.path_cost=path_cost
    def getState(self):
        return self.state
    def getParent(self):
        return self.parent
    def getAction(self):
        return self.action
    def getPathCost(self):
        return self.path_cost

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
"""
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"
    #for  successor in problem.getSuccessors(problem.getStartState()):
        #(nextState,action,cost)=successor
        #print("Urmatoarea stare posibila este: ", nextState," ,in directia:",action," cu costul:",cost)
    # from game import Directions
    # w = Directions.WEST
    # return [w, w]
    # successor=problem.getSuccessors(problem.getStartState())[0]
    # (nextState,action,cost)=successor
    # successor1=problem.getSuccessors(nextState)[0]
    # (nextState1, action1, cost1) = successor1
    # print("Cele doua actiuni sunt:",action," si", action1)
    #from util import Stack
    #node1=CustomNode("first",2)
    #node2=CustomNode("second",4)

    #my_stack=Stack()
    #my_stack.push(node1)
    #my_stack.push(node2)
    #pop_elem=my_stack.pop()
    #print(pop_elem.getName())

    from util import Stack
    myStack = Stack()
    curent = problem.getStartState()
    visited = []
    node = Node(curent, None, None, None)
    myStack.push((node, []))

    while not myStack.isEmpty():

        curentNode, solution = myStack.pop()
        curent = curentNode.getState()
        visited.append(curent)

        if problem.isGoalState(curent):
            return solution

        successors = problem.getSuccessors(curent)
        for i in successors:
            if not i[0] in visited:
                node = Node(i[0], curent, i[1], i[2])
                myStack.push((node, solution + [i[1]]))

    return []

def breadthFirstSearch(problem):
    curent = problem.getStartState()
    if problem.isGoalState(curent):
        return []

    myQueue = util.Queue()
    visited = []
    # (node,actions)
    nod1=Node(curent,None,None,None)
    myQueue.push((nod1, []))

    while not myQueue.isEmpty():
        qNode, solution = myQueue.pop()
        currentNode=qNode.getState()
        if currentNode not in visited:
            visited.append(currentNode)

            if problem.isGoalState(currentNode):
                return solution

            for nextNode, action, cost in problem.getSuccessors(currentNode):
                node=Node(nextNode,currentNode,action,cost)
                newAction = solution + [action]
                myQueue.push((node, newAction))


def uniformCostSearch(problem):
    start = problem.getStartState()
    q = util.PriorityQueue()
    visited = []

   # q.push((start, [], 0), 0)
    node = Node(start,None,[],0)
    q.push(node,0)

    while not q.isEmpty():

       # currentNode, actions, cost = q.pop()
        currentNode = q.pop()
        if currentNode.getState() not in visited:
            visited.append(currentNode.getState())

            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for nextNode, action, node_cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getPathCost() + node_cost
                newNode = Node(nextNode,currentNode,newAction,newCost)

                q.push(newNode, newCost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    q = util.PriorityQueue()
    visited = []

   # q.push((start, [], 0), 0)
    node = Node(start,None,[],0)
    q.push(node,0)

    while not q.isEmpty():

       # currentNode, actions, cost = q.pop()
        currentNode = q.pop()
        if currentNode.getState() not in visited:
            visited.append(currentNode.getState())

            if problem.isGoalState(currentNode.getState()):
                return currentNode.getAction()

            for nextNode, action, node_cost in problem.getSuccessors(currentNode.getState()):
                newAction = currentNode.getAction() + [action]
                newCost = currentNode.getPathCost() + node_cost
                heuristicCost = newCost + heuristic(nextNode, problem)
                newNode = Node(nextNode,currentNode,newAction,newCost)
                q.push(newNode, heuristicCost)

    util.raiseNotDefined()



def randomSearch(problem):
    import random
    solution=[]
    current=problem.getSuccessors(problem.getStartState())
    while not problem.isGoalState(current):
        successors = problem.getSuccessors(current)
        random_index=random.randint(0,len(successors)-1)
        next_state=successors[random_index]
        action=next_state[1]
        solution.append(action)
        current=next_state[0]

    print("Solution: ", solution)
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
