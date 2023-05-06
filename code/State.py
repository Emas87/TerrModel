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
            2: 'd', 
            3: 'a', 
            4: 'attack',
            5: 'cut'
        }

        self.classes = classes
        self.inventory = inventory
        self.map = map
    
    def run_action(self, action):
        new_state = State()

        # Get current health
        health = self.map.getHealth()
        reward = 0
        if action == 1: # space
            if self.map.canJump():
                # Move map downward
                self.map.moveMap(new_state.map.current_map, new_i=1)
                reward = 1
            pass
        elif action == 2: # d
            if self.map.canMove(right=True):
                # Move map left
                self.moveMap(new_state.map.current_map, new_j=-1)
                reward = 1
        elif action == 3: # a
            if self.map.canMove(right=False):
                self.map.moveMap(new_state.map.current_map, new_j=1)
                # Move map left
                reward = 1
            pass
        elif action == 4: # attack
            # find closest enemy position and check 
            with open("delete.txt", 'w') as f:
                f.write(str(self.map))
            attack, x, y = self.map.isEnemyOnAttackRange()
            # if is in attack range
            if attack:
                # delete slime tiles from map accroding to x and y
                self.map.deleteTileAt(x, y, self.classes.index('slime'))
                reward = 2
            else:
                # if not
                reward = 0
        elif action == 5: # cut wood
            # find closest tree position and check 
            # if is in cut range
            cut, x, y = self.map.isTreeOnCutRange()
            # if is in attack range
            if cut:
                # delete tree tiles from map accroding to x and y
                self.map.deleteTileAt(x, y, self.classes.index('tree'))       
                reward = 3
            else:
                # if not
                reward = 0

        new_state.last_action = action
        # Apply effects of the map, gravity attack by an enemy
        new_state.map.fixMap(action == 1)

        new_health = new_state.map.getHealth()
        if new_health - health < 0:
            reward = reward - 3

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
        actions = [0]
        if self.map.canJump():
            actions.append(1)
        if self.map.canMove(right=True):
            actions.append(2)
        if self.map.canMove(right=False):
            actions.append(3)
        if self.map.isEnemyOnAttackRange()[0]:
            actions.append(4)
        if self.map.isTreeOnCutRange()[0]:
            actions.append(5)
        return actions       


if __name__ == "__main__":
    env = TerrEnv()
    state = State()
    obs = env.reset()
    state.map = obs['map']
    state.inventory = obs['inventory']

    while True:
        action = input("Enter action: ")
        if action == 'q':
            break
        with open("old_state.txt", 'w') as f:
            f.write(str(state.map))
        state, reward = state.run_action(int(action))
        with open("old_state.txt", 'w') as f:
            f.write(str(state.map))
  
