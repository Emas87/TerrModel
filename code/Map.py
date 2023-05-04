
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
