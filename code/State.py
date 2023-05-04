import os
from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
import time
import gym
from gym import spaces
from TerrarianEyes import TerrarianEyes
from Inventory import Inventory
from Map import Map
from TerrEnv import TerrEnv

class State():
    def __init__(self):
        self.last_action = None
        self.action_map = {
            0: 'no_op',
            1: 'space',
            2: 'w', 
            3: 'd', 
            4: 'a', 
            5: 'attack',
            6: 'cut'
        }

        # Get classes Inventory
        object_classes = []
        with open(self.data_objects, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "names: " in line:
                classesStr = line.replace("names: ",'').replace('\'', '').replace('[','').replace(']', '').replace('\n', '')
                object_classes = classesStr.split(', ')

        tiles_classes = []
        with open(self.data_tiles, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "names: " in line:
                classesStr = line.replace("names: ",'').replace('\'', '').replace('[','').replace(']', '').replace('\n', '')
                tiles_classes = classesStr.split(', ')
        self.classes = tiles_classes + object_classes + ['player', 'xxxx']

        self.inventory = Inventory(self.classes)
        self.map = Map(self.classes)
    
    def __init__(self, map, inventory, classes):
        self.last_action = None
        self.action_map = {
            0: 'no_op',
            1: 'space',
            2: 'w', 
            3: 'd', 
            4: 'a', 
            5: 'attack',
            6: 'cut'
        }

        self.classes = classes
        self.inventory = inventory
        self.map = map
    
    def run_action(self, action):
        new_state = {"map": self.map.current_map, "inventory": self.inventory.inventory}

        # Get current health
        health = self.map.getHealth()
        if action == 0:
            reward = 0
        elif action == 1: # space
            pass
        elif action == 2: # w
            pass
        elif action == 3: # d
            pass
        elif action == 4: # a
            pass
        elif action == 5: # attack
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
        new_health = new_state['map'].map.getHealth()
        if new_health - health < 0:
            reward = reward - 2

        new_state = State(new_state['map'], new_state['inventory'])
        new_state.last_action = action
        return new_state, reward

    def get_current_state(self):
        return {"map": self.map.current_map, "inventory": self.inventory.inventory}
    
    def is_terminal(self,):
        done = False

        health = self.map.getHealth()
        if health <= 0:
            done = True

        # Get win condition certain number of woods
        # TODO

        return done

    def get_available_actions(self, state):
        pass

if __name__ == "__main__":
    env = TerrEnv()
    state = State()
    obs = env.reset()
    state.map = obs['map']
    state.inventory = obs['inventory']

    with open("old_state.txt", 'w') as f:
        f.write(str(state.map))
    state, reward = state.run_action(0)

    with open("old_state.txt", 'w') as f:
        f.write(str(state.map))
  
