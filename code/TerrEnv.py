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
RIGHT = 'd'
LEFT = 'a'

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
            4: 'attack',
            5: 'cut',
            7: 'helmet',
            8: 'breastplate',
            9: 'legs'
        }
        # Create Instance
        weights_path = os.path.join('runs', 'train', 'yolov5l6', 'weights', 'best.pt')
        self.eyes = TerrarianEyes(weights_path)
        self.timer = None
        self.time_limit = 120
        self.day_timer = time.time()
        self.day_limit = 360
        self.second_phase = False
           
    def step(self, action):
        if self.timer is None:
            raise AssertionError("Cannot call env.step() before calling reset()")
        print(f"Step: {action}")
        
        # More information
        info = {}
        # Get current health
        health = self.eyes.map.getHealth()
        reward = 0
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
                # Release the left mouse button3
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
            closest = self.eyes.map.getCloserSpiral(self.second_phase)
            # print(f'closest: {closest}')
            if closest[0] - 58 > 0:
                # dRight
                # press '2' key and hold it down
                pydirectinput.keyDown('d')

                # keep the key pressed for 2 seconds
                time.sleep(0.3)
                # release the '2' key
                pydirectinput.keyUp(RIGHT)
                #action = 2
            else:
                # press '3' key and hold it down
                pydirectinput.keyDown(LEFT)

                # keep the key pressed for 2 seconds
                time.sleep(0.3)
                # release the '3' key
                pydirectinput.keyUp(LEFT)
                #action = 3
            if (closest[0] == 0 ) and self.second_phase:
                self.step(10)
            elif(closest[1]<18 or closest[0]<6):
                # workbench is in the inventory, use it
                self.step(10)
        elif action == 7: # Build helmet or use the one in the inventory
            # Check first if helmet is in inventory instead of building it
            helmet, col, row = self.eyes.inventory.getItem('helmet')
            if not helmet: # could be in the build inventory
                # use build inventory to know where to click
                helmet , row = self.eyes.inventory.canBuildHelmet()
            if helmet:
                reward = 8
                if col != -1:
                    # Helmet is in the inventory
                    x, y, w, h = self.eyes.inventory.convertCoords(col, row)

                    # Place it
                    self.drag(int(x+w/2), int(y+h/2), 1840, 500)
                else:
                    # position of the helmet to be built
                    x, y, w, h = self.eyes.inventory.convertCoordsBuild(row)
                    # select helmet in build inventory
                    pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                    pydirectinput.mouseDown(button='left')
                    pydirectinput.mouseUp(button='left')
                    time.sleep(0.5)

                    # click on helmet, drag it to helmet position
                    self.drag(50, 650, 1840, 500)
                # click on 460, 155 to leave any item that couldn't be allocated correctly
                pydirectinput.moveTo(460, 155)
                pydirectinput.mouseDown(button='left')
                pydirectinput.mouseUp(button='left')

        elif action == 8: # Build breastplate or use the one in the inventory
            # Check first if breastplate is in inventory instead of building it
            breastplate, col, row = self.eyes.inventory.getItem('breastplate')
            if not breastplate: # could be in the build inventory
                # use build inventory to know where to click
                breastplate , row = self.eyes.inventory.canBuildBP()
            if breastplate:
                reward = 8
                if col != -1:
                    # breastplate is in the inventory
                    x, y, w, h = self.eyes.inventory.convertCoords(col, row)

                    # Place it
                    self.drag(int(x+w/2), int(y+h/2), 1840, 550)
                else:
                    # position of the breastplate to be built
                    x, y, w, h = self.eyes.inventory.convertCoordsBuild(row)
                    # select breastplate in build inventory
                    pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                    pydirectinput.mouseDown(button='left')
                    pydirectinput.mouseUp(button='left')
                    time.sleep(0.5)

                    # click on breastplate, drag it to breastplate position
                    self.drag(50, 650, 1840, 550)
                # click on 460, 155 to leave any item that couldn't be allocated correctly
                pydirectinput.moveTo(460, 155)
                pydirectinput.mouseDown(button='left')
                pydirectinput.mouseUp(button='left')

        elif action == 9: # Build legs or use the ones in the inventory
            # Check first if legs is in inventory instead of building it
            legs, col, row = self.eyes.inventory.getItem('legs')
            if not legs: # could be in the build inventory
                # use build inventory to know where to click
                legs, row = self.eyes.inventory.canBuildLegs()
            if legs:
                reward = 8
                if col != -1:
                    # legs is in the inventory
                    x, y, w, h = self.eyes.inventory.convertCoords(col, row)

                    # Place it
                    self.drag(int(x+w/2), int(y+h/2), 1840, 600)
                else:
                    # position of the legs to be built
                    x, y, w, h = self.eyes.inventory.convertCoordsBuild(row)
                    # select legs in build inventory
                    pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                    pydirectinput.mouseDown(button='left')
                    pydirectinput.mouseUp(button='left')
                    time.sleep(0.5)

                    # click on legs, drag it to legs position
                    self.drag(50, 650, 1840, 600)
                # click on 460, 155 to leave any item that couldn't be allocated correctly
                pydirectinput.moveTo(460, 155)
                pydirectinput.mouseDown(button='left')
                pydirectinput.mouseUp(button='left')


        elif action == 10: # Build workbench
            # click on workbench, drag it to above a dirt, close to a player, maybe check for slime close
            workbench, col, row = self.eyes.inventory.getItem('workbench')
            if not workbench: # could bein the build inventory
                workbench, row = self.eyes.inventory.getBuild('workbench')
            if workbench or self.second_phase:
                # position of the workbench to be built
                # find right coord to place the workbench
                wb_col, wb_row = self.eyes.map.findWBCoords2Place()
                wb_x, wb_y = self.eyes.map.convertCoords(wb_col, wb_row)
                if wb_x is None or wb_y is None:
                    info["status"] = "NotCompleted"
                else:
                    info["status"] = "Completed"

                    if col != -1:
                        # Workbench is in the inventory
                        # Find workbench in inventory
                        x, y, w, h = self.eyes.inventory.convertCoords(col, row)
                       
                        # select workbench in inventory
                        #pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                        #pydirectinput.mouseDown(button='left')
                        #pydirectinput.mouseUp(button='left')

                        #self.drag(900, 400, wb_x, wb_y)
                        self.drag(int(x+w/2), int(y+h/2), wb_x, wb_y)
                    else:
                        # Find workbench in build inventory
                        x, y, w, h = self.eyes.inventory.convertCoordsBuild(row)

                        # if row is not in the build row, click first in its position and 
                        # then drag it
                        if row != 4:
                            # select workbench in build inventory
                            pydirectinput.moveTo(int(x+w/2), int(y+h/2))
                            pydirectinput.mouseDown(button='left')
                            pydirectinput.mouseUp(button='left')

                        self.drag(50, 650, wb_x, wb_y)

                    # click on 460, 155 to leave any item that couldn't be allocated correctly
                    pydirectinput.moveTo(460, 155)
                    pydirectinput.mouseDown(button='left')
                    pydirectinput.mouseUp(button='left')
                    #print(f'cords for WB: {wb_x}, {wb_y}')


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
        self.eyes.updateMapInventory(raw, print=False)        
        #gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        #resized = cv2.resize(gray, (1920,1080))
        #channel = np.reshape(resized, (1,1080,1920))
        return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}
    
    #def get_objects(self):
    #    raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
    #    self.eyes.updateInventory(raw)
    #    return {"map": self.eyes.map.current_map, "inventory": self.eyes.inventory.inventory}
    
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
        if self.eyes.inventory.inventory[6][2] == self.eyes.inventory.classes.index('legs') and \
            self.eyes.inventory.inventory[6][1] == self.eyes.inventory.classes.index('breastplate') and \
            self.eyes.inventory.inventory[6][0] == self.eyes.inventory.classes.index('helmet'):
            return True
        if not self.second_phase:
            self.secondPhase()
        return False
    
    def secondPhase(self):
        # find wood in inventory and get coordinates
        #self.eyes.inventory.inventory[0][3] = self.eyes.inventory.classes.index('wood')
        wood, col, row = self.eyes.inventory.getItem('wood')
        #print(f'wood: {wood}, col: {col}, row: {row}')
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
                _, _, _, info = self.step(10) 
                #if "status" in info and info["status"] != "NotCompleted":
                self.second_phase = True
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
    #game_env.start(7)
    game_env.reset()
    while not game_env.finished():
        #wb_x, wb_y = game_env.eyes.map.findWBCoords2Place()
        game_env.get_observation()
        game_env.step(7)
        game_env.step(8)
        game_env.step(9)
        


    
   