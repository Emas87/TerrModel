import random
import TerrEnv
import time
import logging
from TerrEnv import TerrEnv
from State import State
from configure_logging import configure_logging

# Configure the shared logger
logger = configure_logging('run_experiments.log')

class RHEA:
    def __init__(self, game_env, horizon, rollouts_per_step):
        self.game_env = game_env  # initialize the game state
        self.action_space = self.game_env.action_map  # the action space
        self.horizon = horizon  # the planning horizon
        self.rollouts_per_step = rollouts_per_step  # number of rollouts per planning step
        self.logger = logger

    def search(self, state, max_time=2):
        start_time = time.time()

        for t in range(self.horizon):
            if time.time() - start_time >= max_time:  # check if 2 seconds have elapsed
                break
            scores = []
            #for a in self.action_space:
            for a in state.get_available_actions():
                if time.time() - start_time >= max_time:  # check if 2 seconds have elapsed
                    break
                score = 0
                for i in range(self.rollouts_per_step):
                    rollout_score = self.rollout(state, a)
                    score += rollout_score
                scores.append((a, score))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            action = scores[0][0]
            #state = self.forward_model(state, action)
            state,_ = state.run_action(action)
        return action

    def rollout(self, state, action):
        #rollout_state = self.forward_model(state, action)
        rollout_state, reward = state.run_action(action)
        #score = 0
        score = reward
        for t in range(self.horizon):
            if rollout_state.is_terminal():
                break
            action = random.choice(self.action_space)
            #rollout_state = self.forward_model(rollout_state, action)
            rollout_state, reward = rollout_state.run_action(action)
            score += reward
        return score
    
    def run(self, seed=0, max_time=2):
        # Setup
        iterations = 4
        state = State()

        # start
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
            #state = State()
            state.cut_tree = 0
            action = self.search(state, max_time)  # get the recommended action
            state.run_action(action)
            self.logger.info(f'Action, selected: {action}')
            self.game_env.step(action)
            iterations += 1
        time2 = time.time()
        self.logger.info(f'time to finish {str(time2-time1)}')
        self.game_env.end()
        return float(time2 - time1)    

if __name__ == "__main__":
    # Setup
    game_env = TerrEnv()
    num_simulations = 1  # number of simulations to run
    iterations = 4
    state = State()
    rhea = RHEA(horizon=2, rollouts_per_step=2)
    rhea.game_env.reset()  # initialize the game state

    # Get number of wood and if it is higher than 100 build
    while not rhea.game_env.finished():
        # get a observation every 4 actions, it takes too much time
        if iterations > 3:
            iterations = 0
            observation = rhea.game_env.get_observation()
            state.map.current_map = observation['map']
            state.inventory.inventory = observation['inventory']
        else:
            observation = rhea.game_env.get_objects()
            state.map.current_map = observation['map']
            state.inventory.inventory = observation['inventory']
        #state = State()
        state.cut_tree = 0
        time1 = time.time()
        action = rhea.search(state, 1)  # get the recommended action
        state.run_action(action)
        time2 = time.time()
        print(f'time to plan action: {str(time2-time1)}, selected:> {action}')
        rhea.game_env.step(action)
        iterations += 1
