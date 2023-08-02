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
            5: 'cut',
            6: 'closer'
        }
        self.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'data.yaml')
        self.cut_tree = 0
        # Get classes Inventory
        classes = []
        with open(self.data, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "names: " in line:
                classesStr = line.replace("names: ",'').replace('\'', '').replace('[','').replace(']', '').replace('\n', '')
                classes = classesStr.split(', ')

        self.classes =  classes + ['player', 'xxxx']

        self.inventory = Inventory(self.classes)
        self.map = Map(self.classes)

        self.score = 0
    
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
            else:
                new_state.map.current_map = self.map.current_map.copy()
        elif action == 2: # d
            if self.map.canMove(right=True):
                # Move map right
                self.map.moveMap(new_state.map.current_map, new_j=-1)
                reward = 2
            else:
                new_state.map.current_map = self.map.current_map.copy()
        elif action == 3: # a
            if self.map.canMove(right=False):
                self.map.moveMap(new_state.map.current_map, new_j=1)
                # Move map left
                reward = 2
            else:
                new_state.map.current_map = self.map.current_map.copy()
        elif action == 4: # attack
            # find closest enemy position and check 
            attack, x, y = self.map.isEnemyOnAttackRange()
            # if is in attack range
            new_state.map.current_map = self.map.current_map.copy()
            if attack:
                # delete tree tiles from map accroding to x and y
                if(self.last_action == 4):
                    new_state.map.deleteTileAt(x, y, self.classes.index('slime'))
                    reward = 2 # to extra to kill a slime
                reward += 11
        elif action == 5: # cut wood
            # find closest tree position and check 
            # if is in cut range
            cut, x, y = self.map.isTreeOnCutRange()
            # if is in attack range
            new_state.map.current_map = self.map.current_map.copy()
            if cut:
                # delete tree tiles, after second cut, from map acording to x and y
                if(self.last_action == 5):
                    new_state.map.deleteTileAt(x, y, self.classes.index('tree'))
                    self.cut_tree += 1    
                    reward = 2 # to extra to cut the tree
                reward += 10
        elif action == 6: # get closer to a tree
            reward = 4
            closest = self.map.getCloser()
            if closest - 58 > 0:
                right = True
                #action = 2
            else:
                right = False
                #action = 3
            if right and self.map.canMove(right):
                # Move map left
                self.map.moveMap(new_state.map.current_map, new_j=-1)
            elif not right and self.map.canMove(right):
                self.map.moveMap(new_state.map.current_map, new_j=1)
        new_state.last_action = action
        # Apply effects of the map, gravity attack by an enemy
        new_state.map.fixMap(action == 1 and self.last_action != 1)

        new_state.map.apply_damage()

        new_health = new_state.map.getHealth()
        if new_health - health < 0:
            reward = reward - 5
        new_state.score = self.score + reward

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

    def get_available_actions(self):
        actions = [0, 6]
        #if self.map.canJump():
        #    actions.append(1)
        if self.map.canMove(right=True):
            actions.append(2)
        if self.map.canMove(right=False):
            actions.append(3)
        if self.map.isEnemyOnAttackRange()[0]:
            actions.append(4)
        if self.map.isTreeOnCutRange()[0]:
            actions.append(5)
        return actions

    def __str__(self) -> str:
        return str(self.map) + str(self.inventory)
    
    def print(self):
        with open("delete.txt", 'w') as f:
            f.write(str(self.map))



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
  
