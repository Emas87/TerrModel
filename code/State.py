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
            2: 'd', 
            3: 'a', 
            4: 'attack',
            5: 'cut'
        }
        self.data_objects = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset_objects','data.yaml')
        self.data_tiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset_tiles', 'data.yaml')
        self.cut_tree = 0
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
    
    def run_action(self, action):
        new_state = State()
        new_state.cut_tree = self.cut_tree

        # Get current health
        health = self.map.getHealth()
        reward = 0
        if action == 0: # no_op
            new_state.map.current_map = self.map.current_map.copy()
        elif action == 1: # space
            if self.map.canJump():
                # Move map downward
                self.map.moveMap(new_state.map.current_map, new_i=1)
                reward = 1
        elif action == 2: # d
            if self.map.canMove(right=True):
                # Move map left
                self.map.moveMap(new_state.map.current_map, new_j=-1)
                reward = 1
            else:
                new_state.map.current_map = self.map.current_map.copy()
                reward = 0
        elif action == 3: # a
            if self.map.canMove(right=False):
                self.map.moveMap(new_state.map.current_map, new_j=1)
                # Move map left
                reward = 1
            else:
                new_state.map.current_map = self.map.current_map.copy()
                reward = 0
        elif action == 4: # attack
            # find closest enemy position and check 
            attack, x, y = self.map.isEnemyOnAttackRange()
            # if is in attack range
            new_state.map.current_map = self.map.current_map.copy()
            if attack:
                # delete slime tiles from map accroding to x and y
                new_state.map.deleteTileAt(x, y, self.classes.index('slime'))      
                reward = 2
            else:
                # if not
                reward = 0
        elif action == 5: # cut wood
            # find closest tree position and check 
            # if is in cut range
            cut, x, y = self.map.isTreeOnCutRange()
            # if is in attack range
            new_state.map.current_map = self.map.current_map.copy()
            if cut:
                # delete tree tiles from map accroding to x and y
                new_state.map.deleteTileAt(x, y, self.classes.index('tree'))
                self.cut_tree += 1    
                reward = 4
            else:
                # if not
                reward = 0

        new_state.last_action = action
        # Apply effects of the map, gravity attack by an enemy
        new_state.map.fixMap(action == 1)

        new_state.map.apply_damage()

        new_health = new_state.map.getHealth()
        if new_health - health < 0:
            reward = reward - 2

        return new_state, reward

    def get_current_state(self):
        return {"map": self.map.current_map, "inventory": self.inventory.inventory}
    
    def is_terminal(self,):
        done = False

        health = self.map.getHealth()
        if health <= 0:
            done = True

        # Get win condition certain number of woods
        if self.cut_tree == 5:
            done = True

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
    #env = TerrEnv()
    state = State()
    #obs = env.reset()
    with open("delete.txt", 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if "Map:" in lines[i]:
            continue
        tiles = lines[i].split(" ")
        for j in range(len(tiles)):
            tile = tiles[j].strip()
            if tile == "hear":
                tile = 'heart'
            if tile == 'play':
                tile = 'player'
            if tile == 'slim':
                tile = 'slime'
            state.map.current_map[i-1][j] = state.classes.index(tile)

    while True:
        action = input("Enter action: ")
        if action == 'q':
            break
        with open("old_state.txt", 'w') as f:
            f.write(str(state.map))
        state, reward = state.run_action(int(action))
        with open("new_state.txt", 'w') as f:
            f.write(str(state.map))
  
