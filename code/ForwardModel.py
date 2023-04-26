from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
# 2. Build the Environment
## 2.1 Create Environment
class TerrEnv(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,1080,1920), dtype=np.uint8)
        self.action_space = Discrete(7)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.done_location = {'top': 460, 'left': 760, 'width': 400, 'height': 70}
        
        
    def step(self, action):
        action_map = {
            0: 'no_op',
            1: 'space',
            2: 'w', 
            3: 's', 
            4: 'd', 
            5: 'a', 
            6: 'left',
            7: '1',
            8: '2',
            9: '3',
            10: '4',
            11: '5',
            12: '6',
            13: '7',
            14: '8',
            15: '9',
            16: 'Esc',
        }
        if action != 0 and action != 6:
            pydirectinput.press(action_map[action])
        elif action == 6:
            pydirectinput.click(500,500)

        done, _ = self.get_done() 
        observation = self.get_observation()
        # calculate real reward
        reward = 1 
        info = {}
        return observation, reward, done, info
        
    
    def reset(self):
        time.sleep(10)
        pydirectinput.click(x=150, y=150)
        return self.get_observation()
        
    def render(self):
        cv2.imshow('Game', self.current_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.close()
         
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1920,1080))
        channel = np.reshape(resized, (1,1080,1920))
        return channel
    
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))
        done_strings = ['You', 'You ']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done, done_cap
# 2.2 Test Environment
env = TerrEnv()
obs=env.get_observation()
plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_GRAY2BGR))
done, done_cap = env.get_done()
plt.imshow(done_cap)
pytesseract.image_to_string(done_cap)[:4]
done
for episode in range(10): 
    obs = env.reset()
    done = False  
    total_reward   = 0
    while not done:
        action = env.action_space.sample()
        print(action)
        obs, reward,  done, info =  env.step(action)
        total_reward  += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))    
# 3. Train the Model
## 3.1 Create Callback
# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker
env_checker.check_env(env)
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
## 3.2 Build DQN and Train
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
env = TerrEnv()
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=10)
model.learn(total_timesteps=100000, callback=callback)
model.load('train_first/best_mode l_50000') 
# 4. Test out Model
for episode in range(5): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)