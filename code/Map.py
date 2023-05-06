
import numpy as np
from math import floor
SLOT_SIZE = (16, 16)
class Map:
    def __init__(self, classes) -> None:
        self.classes = classes
        #self.current_map = [[self.classes.index('x') for j in range(120)] for i in range(67)]
        self.current_map = np.full((67, 120), self.classes.index('xxxx'), dtype=np.int8)
        self.max = (1920, 1080)
        self.min = (0,0)
        self.max_tiles = (120, 67)

    def updateMap(self, x1, y1, w, h, object):
        # If tree reduce width, it shoudl be only 1 tile
        if object == 'tree':
            x1 = x1 + w/4
            w = 15
        for x_tile in range(int(w/SLOT_SIZE[0]) + 1):
            for y_tile in range(int(h/SLOT_SIZE[1]) + 1):
                corner  = (x1 + SLOT_SIZE[0]*x_tile, y1 + SLOT_SIZE[1]*y_tile)
                corner_diff = (corner[0]-self.min[0], corner[1]-self.min[1])
                row = floor(corner_diff[1]/SLOT_SIZE[1] + 0.5) - 1
                col = floor(corner_diff[0]/SLOT_SIZE[0] + 0.5) - 1
                self.current_map[row][col] = self.classes.index(object)
    
    def getHealth(self):
        hearts = 0
        for col in self.current_map[0]:
            if self.classes[col] == 'heart':
                hearts+=1
        hearts = hearts/3
        hearts = hearts + 1
        return hearts

    def __str__(self) -> str:
        str1 = 'Map:\n' + '\n'.join([' '.join([self.classes[item][:4] for item in row]) for row in self.current_map])
        return str1
    
    def isEnemyOnAttackRange(self):
        # Center 960, 540 Coords
        # Center 60 , 33 tiles
        # Attack range Coords x = [920, 1000], y = [500, 580] 
        # Attack range Tiles x = [57, 62], y = [31, 36]

        attack = False
        #Find closest enemy to player 
        player = (60, 33)
        # TODO player should be 58,33
        closest = [0,0]
        min_distance = float('inf')
        debug_matrix = [['x' for j in range(57,60 + 1)] for i in range(30,34 + 1)]
        for i in range(30,34 + 1):
            for j in range(57, 60 + 1):
                debug_matrix[i-30][j-57] = f'{self.classes[self.current_map[i][j]]} {j} {i}'
                if self.classes[self.current_map[i][j]] == "slime":
                    attack = True
                    distance = abs(i - player[1]) + abs(j - player[0])
                    if distance < min_distance:
                        closest = [i,j]
                        min_distance = distance
                    return attack, closest[1], closest[0]
        return attack, closest[1], closest[0]

    def isTreeOnCutRange(self):
        # Center 960, 540 Coords
        # Center 60 , 33 tiles
        # Cut range Coords x = [900, 1025], y = [490, 590] 
        # Cut range Tiles x = [56, 61], y = [29, 35]

        cut = False
        #Find closest tree to player 
        player = (60, 33)
        closest = [0,0]
        min_distance = float('inf')
        debug_matrix = [['x' for j in range(56,61 + 1)] for i in range(29,35 + 1)]
        for i in range(29,35 + 1):
            for j in range(56,61 + 1):
                debug_matrix[i-29][j-56] = f'{self.classes[self.current_map[i][j]]} {j} {i}'
                if self.classes[self.current_map[i][j]] == "tree":
                    distance = abs(i - player[1]) + abs(j - player[0])
                    if distance < min_distance:
                        cut = True
                        closest = [i,j]
                        min_distance = distance
        return cut, closest[1], closest[0]

    def canJump(self):
        tile_above = self.classes[self.current_map[32][58]] == 'dirt' or self.classes[self.current_map[32][59]] == 'dirt'
        is_grounded = self.classes[self.current_map[36][58]] == 'dirt' or self.classes[self.current_map[36][59]] == 'dirt' 
        return is_grounded and not tile_above

    def canMove(self, right=True):
        if right:
            tiles_right = self.classes[self.current_map[33][60]] == 'dirt' or self.classes[self.current_map[34][60]] == 'dirt'
            return not tiles_right
        else:
            tiles_left = self.classes[self.current_map[33][57]] == 'dirt' or self.classes[self.current_map[34][57]] == 'dirt'
            return not tiles_left

    def deleteTileAt(self, x, y, clss):
        # delete recursively
        if x < 0 or y < 0:
            return
        if self.current_map[y][x] != clss:
            return
        self.current_map[y][x] = 'xxxx'
        siblings = [(x+1,y), (x,y+1), (x+1,y+1), (x-1,y), (x,y-1), (x-1,y-1), (x+1, y-1), (x-1, y+1) ]
        for sibling in siblings:
            self.deleteEnemyAt(sibling[0], sibling[1], clss)

    def moveMap(self, new_map, new_i=0, new_j=0):
        if new_j > 0 and new_i == 0 and self.classes[self.current_map[35][60]] == 'dirt':
            # check if there is a small tile that player con walk above 
            new_i = 1
        elif new_j < 0 and new_i == 0 and self.classes[self.current_map[35][57]] == 'dirt':
            new_i = 1

        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                # Do not move player
                if (i == 33 and j == 58) or (i == 34 and j == 58) or (i == 35 and j == 58) or \
                   (i == 33 and j == 59) or (i == 34 and j == 59) or (i == 35 and j == 59):
                    continue
                if i+new_i < len(new_map) and j+new_j < len(new_map[i+new_i]) and i+new_i >= 0 and j+new_j >= 0:
                    new_map[i+new_i][j + new_j] = self.current_map[i][j]
        # player is always in same position
        new_map[33][58] = self.classes.index('player')
        new_map[34][58] = self.classes.index('player')
        new_map[35][58] = self.classes.index('player')
        new_map[33][59] = self.classes.index('player')
        new_map[34][59] = self.classes.index('player')
        new_map[35][59] = self.classes.index('player')
    
    def fixMap(self, jump=False):
        if not jump:
            # fix player in the air
            new_map = np.full((67, 120), self.classes.index('xxxx'), dtype=np.int8)
            self.moveMap(new_map, new_i=-1)
            self.current_map = new_map
        
        # fix any floating tree left to player
        found_tree = False
        for i  in range(67):
            if self.classes[self.current_map[i][57]] == 'tree':
                found_tree = True
            elif found_tree and self.classes[self.current_map[i][57]] == 'xxxx':
                self.current_map[i][57] = self.classes.index('tree')
            elif self.classes[self.current_map[i][58]] == 'dirt':
                break

        found_tree = False
        # fix any floating tree right to player
        for i  in range(67):
            if self.classes[self.current_map[i][60]] == 'tree':
                found_tree = True
            elif found_tree and self.classes[self.current_map[i][60]] == 'xxxx':
                self.current_map[i][60] = self.classes.index('tree')
            elif self.classes[self.current_map[i][58]] == 'dirt':
                break

        found_tree = False
        # fix floating tree above player
        for i  in range(67):
            if self.classes[self.current_map[i][58]] == 'tree':
                found_tree = True
            elif found_tree and self.classes[self.current_map[i][58]] == 'xxxx':
                self.current_map[i][58] = self.classes.index('tree')
            elif self.classes[self.current_map[i][58]] == 'player':
                break
        found_tree = False
        # fix floating tree above player
        for i  in range(67):
            if self.classes[self.current_map[i][59]] == 'tree':
                found_tree = True
            elif found_tree and self.classes[self.current_map[i][59]] == 'xxxx':
                self.current_map[i][58] = self.classes.index('tree')
            elif self.classes[self.current_map[i][59]] == 'player':
                break

