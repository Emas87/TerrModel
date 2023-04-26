import random
import TerrEnv

class RHEA:
    def __init__(self, forward_model, action_space, horizon, rollouts_per_step):
        self.forward_model = forward_model  # the forward model function
        self.action_space = action_space  # the action space
        self.horizon = horizon  # the planning horizon
        self.rollouts_per_step = rollouts_per_step  # number of rollouts per planning step

    def search(self, state):
        for t in range(self.horizon):
            scores = []
            for a in self.action_space:
                score = 0
                for i in range(self.rollouts_per_step):
                    rollout_score = self.rollout(state, a)
                    score += rollout_score
                scores.append((a, score))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            action = scores[0][0]
            state = self.forward_model(state, action)
        return action

    def rollout(self, state, action):
        rollout_state = self.forward_model(state, action)
        score = 0
        for t in range(self.horizon):
            if rollout_state.is_terminal():
                break
            action = random.choice(self.action_space)
            rollout_state = self.forward_model(rollout_state, action)
            score += rollout_state.reward()
        return score
    
game_state = TerrEnv()  # initialize the game state

rhea = RHEA(forward_model=None, action_space=game_state.action_map, horizon=10, rollouts_per_step=100)
game_state.reset()  # initialize the game state
action = rhea.search(game_state)  # get the recommended action
