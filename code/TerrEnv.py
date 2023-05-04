import os
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
import gym
from gym import spaces
from TerrarianEyes import TerrarianEyes

class TerrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space = spaces.Dict(
            {
                'map' :       spaces.Box(low=0, high=11, shape=(67, 120), dtype=np.int8),
                'inventory' : spaces.Box(low=0, high=11, shape=(9,10), dtype=np.int8),
            }
        )
        self.action_space = spaces.Discrete(7)
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
            5: 'attack',
            6: 'cut'
        }
        # Create Instance
        tiles_weights_path = os.path.join('runs', 'train', 'yolov5s6-tiles', 'weights', 'best.pt')
        objects_weights_path = os.path.join('runs', 'train', 'yolov5l6-objects', 'weights', 'best.pt')
        self.eyes = TerrarianEyes(tiles_weights_path, objects_weights_path)
        self.timer = None
        self.time_limit = 120
        self.day_timer = time.time()
        self.day_limit = 360
           
    def step(self, action):
        if self.timer is None:
            raise AssertionError("Cannot call env.step() before calling reset()")
        # Get current health
        health = self.eyes.map.getHealth()
        if action == 0:
            reward = 0
        elif action < 5:
            pydirectinput.press(self.action_map[action])
            reward = 1 
        else: 
            # In case we need map or inventory
            if action == 5: # attack
                # find closest enemy position and check 
                #with open("delete.txt", 'w') as f:
                #    f.write(str(self.eyes.map))
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
            elif action == 6: # cut wood
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

        done, _ = self.get_done() 
        observation = self.get_observation()
        new_health = self.eyes.map.getHealth()
        if new_health - health < 0:
            reward = reward - 2
        # calculate real reward
        info = {}
        return observation, reward, done, info
        
    
    def reset(self):
        time.sleep(10)
        pydirectinput.click(x=150, y=250)
        self.timer = time.time()
        observation = self._get_obs()
        return observation
        
    def render(self):
        cv2.imshow('Game', self.current_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.close()
         
    def close(self):
        cv2.destroyAllWindows()
    
    #def _get_obs(self):
        #return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}

    def _get_obs(self):
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        self.eyes.updateMap(raw)
        self.eyes.updateInventory(raw)
        #gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        #resized = cv2.resize(gray, (1920,1080))
        #channel = np.reshape(resized, (1,1080,1920))
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}
    
    def get_done(self):
        done = False
        done_cap = None
        if time.time() - self.day_timer  > self.day_limit:
            # reset day

            # Open inventory
            pydirectinput.press('Esc')
            # Click Power menu
            pydirectinput.moveTo(50, 315)
            # Press the left mouse button
            pydirectinput.mouseDown(button='left')
            # Release the left mouse button
            pydirectinput.mouseUp(button='left')

            #Click other place in the power menu to reset view
            pydirectinput.moveTo(50, 530)
            # Press the left mouse button
            pydirectinput.mouseDown(button='left')
            # Release the left mouse button
            pydirectinput.mouseUp(button='left')

            #Click time menu
            pydirectinput.moveTo(50, 630)
            # Press the left mouse button
            pydirectinput.mouseDown(button='left')
            # Release the left mouse button
            pydirectinput.mouseUp(button='left')

            #Click dawn
            pydirectinput.moveTo(115, 655)
            # Press the left mouse button
            pydirectinput.mouseDown(button='left')
            # Release the left mouse button
            pydirectinput.mouseUp(button='left')

            
            # Close inventory
            pydirectinput.press('Esc')

            self.day_timer = time.time()            

        elif time.time() - self.timer  > self.time_limit:
            done = True
        else:
            done_cap = np.array(self.cap.grab(self.done_location))
            done_strings = ['You', 'You ']
            res = pytesseract.image_to_string(done_cap)[:4]
            if res in done_strings:
                done = True
        return done, done_cap