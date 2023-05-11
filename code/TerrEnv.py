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
import logging
from configure_logging import configure_logging

# Configure the shared logger
logger = configure_logging('run_experiments.log')

class TerrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.win = False
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
            2: 'd', 
            3: 'a', 
            4: 'attack',
            5: 'cut'
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
        reward = 0
        if action < 4:
            # press 'action' key and hold it down
            pydirectinput.keyDown(self.action_map[action])

            # keep the key pressed for 2 seconds
            time.sleep(0.3)
            # release the 'action' key
            pydirectinput.keyUp(self.action_map[action])
            reward = 1 
        else: 
            # In case we need map or inventory
            if action == 4: # attack
                # find closest enemy position and check 
                #with open("delete.txt", 'w') as f:
                #    f.write(str(self.eyes.map))
                attack, x, y= self.eyes.map.isEnemyOnAttackRange()
                # if is in attack range
                if attack:
                    # Move mouse to enemy position
                    pydirectinput.moveTo((x+1)*16 + 16, y*16 + 8)
                    # attack
                    pydirectinput.press('3')
                    # Press the left mouse button
                    pydirectinput.mouseDown(button='left')
                    time.sleep(6)
                    # Release the left mouse button
                    pydirectinput.mouseUp(button='left')
                    reward = 20
            elif action == 5: # cut wood
                # find closest tree position and check 
                # if is in cut range
                cut, x, y = self.eyes.map.isTreeOnCutRange()
                # if is in attack range
                if cut:
                    # Move mouse to enemy position
                    pydirectinput.moveTo((x+1)*16 + 16, y*16 + 8)
                    
                    # attack
                    pydirectinput.press('3')
                    # Press the left mouse button
                    pydirectinput.mouseDown(button='left')
                    time.sleep(2)
                    # Release the left mouse button3
                    pydirectinput.mouseUp(button='left')
                    reward = 10
            elif action == 6: # closer
                reward = 4
                closest = self.eyes.map.getCloser()
                if closest - 58 > 0:
                    # press '2' key and hold it down
                    pydirectinput.keyDown(self.action_map[2])

                    # keep the key pressed for 2 seconds
                    time.sleep(0.3)
                    # release the '2' key
                    pydirectinput.keyUp(self.action_map[2])
                    #action = 2
                else:
                    # press '3' key and hold it down
                    pydirectinput.keyDown(self.action_map[3])

                    # keep the key pressed for 2 seconds
                    time.sleep(0.3)
                    # release the '3' key
                    pydirectinput.keyUp(self.action_map[3])
                    #action = 3
            elif action == 10: # Build
                # click on workbench, drag it to above a dirt, close to a player, maybe check for slime close
                workbench , row = self.eyes.inventory.getBuild('workbench')
                if workbench:
                    x, y, w, h = self.eyes.inventory.convertCoordsBuild(row)
                    # select workbench
                    pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                    pydirectinput.mouseDown(button='left')
                    pydirectinput.mouseUp(button='left')

                    # position of the item to be build
                    self.drag(50, 650,965,555)
                    # TODO
                    # try to find helmet
                    # click on helmet, dragit to helmet position
                    # click below helmet in build menu, drag it to breastplate possition
                    # click below breastplate in build menu, drag it to legs possition
                    # set win to True
                    self.win = True
                    reward = 100
        done= self.get_done() 
        died= self.died() 
        if died:
            observation = self.reset()
            reward = reward - 5
        else:
            observation = self.get_observation()
            new_health = self.eyes.map.getHealth()
            if new_health - health < 0:
                reward = reward - 2
        # calculate real reward
        info = {}
        return observation, reward, done, info
        
    def reset(self):
        self.win = False
        time.sleep(10)
        #pydirectinput.click(x=150, y=250)
        # Press Esc
        pydirectinput.keyDown('esc')
        time.sleep(0.2)
        pydirectinput.keyUp('esc')
        self.timer = time.time()
        observation = self.get_observation()
        return observation
    
    @staticmethod
    def drag(x1,y1,x2,y2):
        pydirectinput.moveTo(x1, y1)
        pydirectinput.mouseDown(button='left')
        pydirectinput.mouseUp(button='left')

        pydirectinput.moveTo(x2, y2)
        pydirectinput.mouseDown(button='left')
        pydirectinput.mouseUp(button='left')               
        #time.sleep(0.5)
  
    def render(self):
        cv2.imshow('Game', self.current_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.close()
         
    def close(self):
        cv2.destroyAllWindows()
    
    def _get_obs(self):
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        self.eyes.updateMap(raw)
        self.eyes.updateInventory(raw)
        #with open("delete.txt", 'w') as f:
        #    f.write(str(self.eyes.map))
        #gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        #resized = cv2.resize(gray, (1920,1080))
        #channel = np.reshape(resized, (1,1080,1920))
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}
    
    def get_objects(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        self.eyes.updateInventory(raw)
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}
    
    def get_done(self):
        if self.win:
            return True
        
    def died(self):
        done = False        
        done_cap = np.array(self.cap.grab(self.done_location))
        done_strings = ['You', 'You ']
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        return done
    
    def finished(self):
        # find wood in inventory and get coordinates
        #self.eyes.inventory.inventory[0][3] = self.eyes.inventory.classes.index('wood')
        wood, col, row = self.eyes.inventory.getWood()
        # find number
        if wood:
            x, y, w, h = self.eyes.inventory.convertCoords(col, row)
            #location = {'top': x, 'left': int(y + h/2), 'width': w, 'height': int(h/2)}
            # raw = np.array(self.cap.grab(location))[:,:,:3].astype(np.uint8)
            location = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
            raw = np.array(self.cap.grab(location))[:,:,:3].astype(np.uint8)
            count = self.eyes.findNumber(raw, x, int(y + h/2), w, int(h/2))
            # is bigger than 100
            self.logger.info(f"wood: {count}")
            if count >= 100 and count < 200:
                # build
                self.step(10) 
                return True
        return False
    
    def start(self, seed):
        # click 345, 435 Open Terraria
        pydirectinput.click(345, 435, clicks = 2)

        # wait until is opened
        time.sleep(25)

        # Click 'Single Palyer' 945, 300
        pydirectinput.moveTo(945, 300)
                    
        pydirectinput.mouseDown(button='left')
        time.sleep(0.5)
        pydirectinput.mouseUp(button='left')

        # Click Top Player 945, 300
        pydirectinput.mouseDown(button='left')
        time.sleep(0.2)
        pydirectinput.mouseUp(button='left')
        pydirectinput.mouseDown(button='left')
        time.sleep(0.2)
        pydirectinput.mouseUp(button='left')

        if seed == 0:
            pydirectinput.moveTo(945, 345)
        elif seed == 1:
            pydirectinput.moveTo(945, 460)
        elif seed == 2:
            pydirectinput.moveTo(945, 575)
        elif seed == 3:
            pydirectinput.moveTo(945, 690)
        elif seed == 4:
            pydirectinput.moveTo(945, 805)
        elif seed == 5:
            # scroll mouse
            pydirectinput.moveTo(1323,546)
            pydirectinput.mouseDown(button='left')
            time.sleep(0.2)
            pydirectinput.mouseUp(button='left')
            pydirectinput.moveTo(945, 345)
        elif seed == 6:
            pydirectinput.moveTo(1323,546)
            pydirectinput.mouseDown(button='left')
            time.sleep(0.2)
            pydirectinput.mouseUp(button='left')
            pydirectinput.moveTo(945, 460)
        elif seed == 7:
            pydirectinput.moveTo(1323,546)
            pydirectinput.mouseDown(button='left')
            time.sleep(0.2)
            pydirectinput.mouseUp(button='left')
            pydirectinput.moveTo(945, 575)
        
        pydirectinput.mouseDown(button='left')
        time.sleep(0.2)
        pydirectinput.mouseUp(button='left')
        pydirectinput.mouseDown(button='left')
        time.sleep(0.2)
        pydirectinput.mouseUp(button='left')
        time.sleep(3)
        pydirectinput.click(x=930, y=512)        

        # delete any rest of wood in case player died in previous experiments
        pydirectinput.keyDown('esc')
        time.sleep(0.2)
        pydirectinput.keyUp('esc')

        pydirectinput.keyDown('ctrl')
        pydirectinput.moveTo(x=205, y=45)
        pydirectinput.mouseDown(button='left')
        time.sleep(0.1)
        pydirectinput.mouseUp(button='left')

        pydirectinput.moveTo(x=255, y=45)
        pydirectinput.mouseDown(button='left')
        time.sleep(0.1)
        pydirectinput.mouseUp(button='left')

        pydirectinput.moveTo(x=305, y=45)        
        pydirectinput.mouseDown(button='left')
        time.sleep(0.1)
        pydirectinput.mouseUp(button='left')

        pydirectinput.keyUp('ctrl')

        pydirectinput.keyDown('esc')
        time.sleep(0.2)
        pydirectinput.keyUp('esc')

        pydirectinput.press('ctrl')   


    def end(self):
        pydirectinput.keyDown('alt')
        pydirectinput.press('f4')
        pydirectinput.keyUp('alt')
        time.sleep(2)

if __name__ == "__main__":
    
    # Setup
    
    game_env = TerrEnv()  # initialize the game state
    #game_env.reset()  # initialize the game state
    game_env.start(7)
    game_env.reset()
    while True:
        game_env.get_observation()
        game_env.finished()


    
   