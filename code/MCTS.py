import math
import random
import time
from TerrEnv import TerrEnv
from State import State

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.wins += result
        self.visits += 1

    def fully_expanded(self):
        actions = self.state.get_available_actions()
        return len(self.children) == len(actions), actions

    def best_child(self, exploration):
        def ucb(node):
            return node.wins / node.visits + exploration * math.sqrt(math.log(self.visits) / node.visits)
        return max(self.children, key=ucb)

class MCTS:
    def __init__(self, exploration=1.0):
        self.exploration = exploration
        self.game_env = TerrEnv()  # initialize the game state


    def select(self, root_node):
        node = root_node
        fully = False
        while not node.state.is_terminal():
            full, actions = node.fully_expanded()
            if not full:
                return self.expand(node, actions),  fully
            else:
                node = node.best_child(self.exploration)
                fully = True
        return node, fully

    def expand(self, node, actions):
        #actions = node.state.get_available_actions()
        untried_actions = [action for action in actions if not any(child.state.last_action == action for child in node.children)]
        if untried_actions:
            action = random.choice(untried_actions)
            child_state, _ = node.state.run_action(action)
            child_node = node.add_child(child_state)
            return child_node
        else:
            child_node = random.choice(node.children)
            return child_node

    def simulate(self, state, max_iterations=None):
        iterations = 0
        actions = []
        #result = 0
        while not state.is_terminal() and ((max_iterations is None) or (iterations < max_iterations)):
            action = random.choice(state.get_available_actions())
            actions.append(action)
            state, _ = state.run_action(action)
            #result += reward
            iterations += 1
        return state.score, actions
        #return result, actions

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def search(self, state, max_time=40, max_iterations=None):
        root_node = Node(state)
        start_time = time.time()
        while not state.is_terminal():
            node, fully_expanded = self.select(root_node)
            #if fully_expanded:
            #    break
            result, actions = self.simulate(node.state, max_iterations)
            print(f"simulate with {node.state.last_action}, result: {result}, with these actions {str(actions)}, initial score {node.state.score}")
            self.backpropagate(node, result)
            if time.time() - start_time >= max_time:
                break
        if len(root_node.children) == 0:
            return 0
        action =  max(root_node.children, key=lambda node: node.wins).state.last_action
        #action =  max(root_node.children, key=lambda node: node.state.score).state.last_action
        if action == 6:
            print()
            pass        
        return action

    def run(self, seed, max_time=1):
        # Setup
        iterations = 4
        state = State()
        num_simulations = 1  # number of simulations to run

        self.game_env.start(seed)
        self.game_env.reset()  # initialize the game state
        time1 = time.time()

        # Get number of wood and if it is higher than 100 build
        while not self.game_env.finished():
            # get a observation every 4 actions, it takes too much time
            if iterations > 3:
                iterations = 0
                observation = self.game_env.get_observation()
                state.map.current_map = observation['map']
                state.inventory.inventory = observation['inventory']
            else:
                observation = self.game_env.get_objects()
                state.map.current_map = observation['map']
                state.inventory.inventory = observation['inventory']
            state.cut_tree = 0
            action = self.search(state, max_iterations=num_simulations, max_time=max_time)  # get the recommended action
            state.run_action(action)
            self.game_env.step(action)
            iterations += 1
        time2 = time.time()
        print(f'time to finish {str(time2-time1)}')
        return float(time2 - time1)

if __name__ == "__main__":
    # Setup
    mcts = MCTS(exploration=3)
    mcts.game_env.reset()  # initialize the game state
    num_simulations = 1  # number of simulations to run
    iterations = 4
    state = State()

    # Get number of wood and if it is higher than 100 build
    while not mcts.game_env.finished():
        # get a observation every 4 actions, it takes too much time
        if iterations > 3:
            iterations = 0
            observation = mcts.game_env.get_observation()
            state.map.current_map = observation['map']
            state.inventory.inventory = observation['inventory']
        else:
            observation = mcts.game_env.get_objects()
            state.map.current_map = observation['map']
            state.inventory.inventory = observation['inventory']
        #state = State()
        state.cut_tree = 0
        time1 = time.time()
        action = mcts.search(state, max_iterations=num_simulations, max_time=1)  # get the recommended action
        state.run_action(action)
        time2 = time.time()
        print(f'time to plan action: {str(time2-time1)}, selected:> {action}')
        mcts.game_env.step(action)
        iterations += 1
    mcts.game_env.end()
