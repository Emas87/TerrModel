import os
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from TerrarianEyes import TerrarianEyes

class TerrEnv(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,1080,1920), dtype=np.uint8)
        self.action_space = Discrete(8)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.done_location = {'top': 460, 'left': 760, 'width': 400, 'height': 70}
        self.action_map = {
            0: 'no_op',
            1: 'space',
            2: 'w', 
            3: 'd', 
            4: 'a', 
            5: 'h', # heal
            6: 'attack',
            7: 'cut',
            #8: 'mine up',
            #9: 'mine down',
            #10: 'mine left',
            #11: 'mine right',
            #12: 'attack bow',
            #13: 'use torch',
            #14: 'use rope',
            #15: 'Esc', # open inventory
        }
        # Create Instance
        tiles_weights_path = os.path.join('runs', 'train', 'yolov5l6-tiles', 'weights', 'best.pt')
        objects_weights_path = os.path.join('runs', 'train', 'yolov5l6-objects', 'weights', 'best.pt')
        self.eyes = TerrarianEyes(tiles_weights_path, objects_weights_path)
        self.observation = None
        self.timer = None
        self.time_limit = 120
           
    def step(self, action):
        self.eyes.updateMap(self.observation)
        # Get current health
        health = self.eyes.map.getHealth()
        if action == 0:
            reward = 0
        elif action < 6:
            pydirectinput.press(self.action_map[action])
            reward = 1 
        else: 
            self.eyes.updateInventory(self.observation)
            # In case we need map or inventory
            if action == 6: # attack
                # find closest enemy position and check 
                with open("delete.txt", 'w') as f:
                    f.write(str(self.eyes.map))
                attack, x, y= self.eyes.map.isEnemyOnAttackRange()
                # if is in attack range
                if attack:
                    # Move mouse to enemy position
                    pydirectinput.moveTo((x+1)*16 + 16, y*16 + 8)
                    # attack
                    pydirectinput.press('1')
                    # Press the left mouse button
                    pydirectinput.mouseDown(button='left')
                    # Release the left mouse button
                    pydirectinput.mouseUp(button='left')
                    reward = 2
                else:
                    # if not
                    reward = 0
            elif action == 7: # cut wood
                # find closest tree position and check 
                # if is in cut range
                cut, x, y = self.eyes.map.isTreeOnCutRange()
                # if is in attack range
                if cut:
                    # Move mouse to enemy position
                    pydirectinput.moveTo((x+1)*16 + 8, y*16 + 8)
                    
                    # attack
                    pydirectinput.press('3')
                    # Press the left mouse button
                    pydirectinput.mouseDown(button='left')
                    # Release the left mouse button
                    pydirectinput.mouseUp(button='left')
                    reward = 3
                else:
                    # if not
                    reward = 0

            elif action == 8: # mine up FUTURE WORK
                # Move mouse above player position
                pydirectinput.press(self.action_map[2])
                # Check what was mined
                reward = 2
            elif action == 9: # mine down FUTURE WORK
                # Move mouse below player position
                pydirectinput.press(self.action_map[2])
                # Check what was mined
                reward = 2
            elif action == 10: # mine left FUTURE WORK
                # Move mouse left to player position
                pydirectinput.press(self.action_map[2])
                # Check what was mined
                reward = 2
            elif action == 11: # mine right FUTURE WORK
                # Move mouse right to player position
                pydirectinput.press(self.action_map[2])
                # Check what was mined
                reward = 2
            elif action == 12:
                # Find bow and use it in closest enemy FUTURE WORK
                pydirectinput.press(self.action_map[4])
            elif action == 13:
                # Find torch and use it FUTURE WORK
                pydirectinput.press(self.action_map[5])
            elif action == 14:
                # Find rope and use it FUTURE WORK
                pydirectinput.press(self.action_map[6])

        done, _ = self.get_done() 
        observation = self.get_observation()
        # calculate real reward
        info = {}
        return observation, reward, done, info
        
    def reset(self):
        time.sleep(10)
        pydirectinput.click(x=150, y=150)
        self.timer = time.time()
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
        self.observation = raw
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1920,1080))
        channel = np.reshape(resized, (1,1080,1920))
        return channel
    
    def get_done(self):
        done = False
        done_cap = None
        if self.timer - time.time() > self.time_limit:
            done = True
        else:
            done_cap = np.array(self.cap.grab(self.done_location))
            done_strings = ['You', 'You ']
            res = pytesseract.image_to_string(done_cap)[:4]
            if res in done_strings:
                done = True
        return done, done_cap
    
env = TerrEnv()
for episode in range(10): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward,  done, info =  env.step(env.action_space.sample())
        total_reward  += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))    
