import math
import random
from TerrEnv import TerrEnv
import time
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
        return len(self.children) == len(self.state.get_available_actions())

    def best_child(self, exploration):
        def ucb(node):
            return node.wins / node.visits + exploration * math.sqrt(math.log(self.visits) / node.visits)
        return max(self.children, key=ucb)

class MCTS:
    def __init__(self, forward_model, exploration=1.0):
        self.forward_model = forward_model
        self.exploration = exploration

    def select(self, root_node):
        node = root_node
        while not node.state.is_terminal():
            if not node.fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child(self.exploration)
        return node

    def expand(self, node):
        actions = node.state.get_available_actions()
        untried_actions = [action for action in actions if not any(child.state.last_action == action for child in node.children)]
        if untried_actions:
            action = random.choice(untried_actions)
            child_state, _ = node.state.run_action(action)
            child_node = node.add_child(child_state)
            return child_node
        else:
            child_node = random.choice(node.children)
            return child_node

    def simulate(self, state):
        while not state.is_terminal():
            action = random.choice(state.get_available_actions())
            state, result = state.run_action(state, action)
        return result

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def search(self, state, max_time=40, max_iterations=None):
        root_node = Node(state)
        start_time = time.time()
        iterations = 0
        while True:
            iterations += 1
            node = self.select(root_node)
            result = self.simulate(node.state)
            self.backpropagate(node, result)
            if time.time() - start_time >= max_time:
                break
            if max_iterations and iterations >= max_iterations:
                break
        return max(root_node.children, key=lambda node: node.visits).state.last_action


if __name__ == "__main__":
    # Setup
    mcts = MCTS(forward_model=State())
    state = State()
    game_env = TerrEnv()  # initialize the game state
    game_env.reset()  # initialize the game state
    num_simulations = 1000  # number of simulations to run

    while not state.is_terminal():
        time1 = time.time()
        action = mcts.search(game_state, max_iterations=num_simulations)  # get the recommended action
        time2 = time.time()
        print(f'time to take action: {str(time2-time1)}')
        game_state.step(action)
