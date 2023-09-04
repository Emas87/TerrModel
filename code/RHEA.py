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
    def __init__(self, game_env, horizon):
        self.game_env = game_env  # initialize the game state
        self.action_space = self.game_env.action_map  # the action space
        self.horizon = horizon  # number of rollouts per planning step
        self.logger = logger

    def search(self, state, max_time=2):
        start_time = time.time()

        scores = []
        available_actions = state.get_available_actions()
        action = available_actions[0]
        print(f'available_actions: {available_actions}')
        if(len(available_actions) == 1):
            return action
        while time.time() - start_time < max_time:
            for a in available_actions:
                if time.time() - start_time >= max_time:  # check if 2 seconds have elapsed
                    break
                score = 0
                for _ in range(self.horizon):
                    rollout_score = self.rollout(state, a)
                    score += rollout_score
                scores.append((a, score))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            print(scores)
            action = scores[0][0]
            #state = self.forward_model(state, action)
            state,_ = state.run_action(action)
            available_actions = state.get_available_actions()
            print(f'available_actions: {available_actions}')
        return action

    def rollout(self, state, action):
        #rollout_state = self.forward_model(state, action)
        rollout_state, reward = state.run_action(action)
        #score = 0
        score = reward
        for t in range(self.horizon):
            if rollout_state.is_terminal():
                break
            action = random.choice(list(self.action_space.keys()))
            #rollout_state = self.forward_model(rollout_state, action)
            rollout_state, reward = rollout_state.run_action(action)
            score += reward
        return score
    
    def run(self, seed=0, max_time=2):
        # Setup
        state = State()

        # start
        self.game_env.start(seed)
        #self.game_env.reset()  # initialize the game state
        time1 = time.time()

        # Get number of wood and if it is higher than 100 build
        while not self.game_env.finished():
            observation = self.game_env.get_observation()
            state.map.current_map = observation['map'].copy()
            state.inventory.inventory = observation['inventory'].copy()
            state.second_phase = self.game_env.second_phase
            action = self.search(state, max_time)  # get the recommended action
            self.logger.info(f'Action, selected: {action}')
            self.game_env.step(action)
        time2 = time.time()
        self.logger.info(f'time to finish {str(time2-time1)}')
        self.game_env.end()
        return float(time2 - time1)    

if __name__ == "__main__":
    # Setup
    game_env = TerrEnv()
    state = State()
    rhea = RHEA(game_env, horizon=2)
    rhea.game_env.reset()  # initialize the game state

    # Get number of wood and if it is higher than 100 build
    while not rhea.game_env.finished():
        # get a observation every 4 actions, it takes too much time
        observation = rhea.game_env.get_observation()
        state.map.current_map = observation['map'].copy()
        state.inventory.inventory = observation['inventory'].copy()
        state.second_phase = rhea.game_env.second_phase
        #time1 = time.time()
        action = rhea.search(state, 2)  # get the recommended action
        #time2 = time.time()
        #print(f'time to plan action: {str(time2-time1)}, selected:> {action}')
        rhea.game_env.step(action)
