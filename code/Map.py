
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
                if row >= 67:
                    row = 66
                if col >= 120:
                    col = 119
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
        player = (60, 32)
        player = (58, 31)
        closest = [0,0]
        min_distance = float('inf')
        debug_matrix = [['x' for j in range(54,63 + 1)] for i in range(27,37 + 1)]
        for i in range(27,37 + 1):
            for j in range(54,63 + 1):
                debug_matrix[i-27][j-54] = f'{self.classes[self.current_map[i][j]]} {j} {i}'
                if self.classes[self.current_map[i][j]] == "slime":
                    distance = abs(i - player[1]) + abs(j - player[0])
                    if distance < min_distance:
                        attack = True
                        closest = [i,j]
                        min_distance = distance
        return attack, closest[1], closest[0]

    def isTreeOnCutRange(self):
        # Center 960, 540 Coords
        # Center 60 , 33 tiles
        # Cut range Coords x = [900, 1025], y = [490, 590] 
        # Cut range Tiles x = [56, 61], y = [29, 35]

        cut = False
        #Find closest tree to player 
        #player = (60, 33)
        # test cut distance from the feet
        player = (58, 33)
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
        tile_above = self.classes[self.current_map[30][58]] == 'dirt' or self.classes[self.current_map[30][59]] == 'dirt'
        is_grounded = self.classes[self.current_map[34][58]] == 'dirt' or self.classes[self.current_map[34][59]] == 'dirt' 
        return is_grounded and not tile_above

    def canMove(self, right=True):
        if right:
            tiles_right = self.classes[self.current_map[31][60]] == 'dirt' or self.classes[self.current_map[32][60]] == 'dirt'
            return not tiles_right
        else:
            tiles_left = self.classes[self.current_map[31][57]] == 'dirt' or self.classes[self.current_map[32][57]] == 'dirt'
            return not tiles_left

    def deleteTileAt(self, x, y, clss):
        # delete recursively
        if x < 0 or y < 0 or x == len(self.current_map) or y == len(self.current_map):
            return
        if self.current_map[y][x] != clss:
            return
        self.current_map[y][x] = self.classes.index('xxxx')
        siblings = [(x+1,y), (x,y+1), (x+1,y+1), (x-1,y), (x,y-1), (x-1,y-1), (x+1, y-1), (x-1, y+1) ]
        for sibling in siblings:
            self.deleteTileAt(sibling[0], sibling[1], clss)

    def moveMap(self, new_map, new_i=0, new_j=0):
        if new_j < 0 and new_i == 0 and self.classes[self.current_map[33][60]] == 'dirt':
            # check if there is a small tile that player con walk above 
            new_i = 1
        elif new_j > 0 and new_i == 0 and self.classes[self.current_map[33][57]] == 'dirt':
            new_i = 1

        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                # Do not move player
                if self.classes.index('player') == self.current_map[i][j] or self.classes.index('heart') == self.current_map[i][j]:
                    new_map[i][j] = self.current_map[i][j]
                    
                elif i+new_i < len(new_map) and j+new_j < len(new_map[i+new_i]) and i+new_i >= 0 and j+new_j >= 0:
                    # do not overwrite player or heart tiles
                    if self.classes.index('player') == new_map[i+new_i][j + new_j] or self.classes.index('heart') == new_map[i+new_i][j + new_j]:
                        continue
                    new_map[i+new_i][j + new_j] = self.current_map[i][j]
    
    def fixSlimes(self, slimes):
        # iterate each one to find if they can move closer to the player,
        for slime in slimes:
            if len(slime) < 4:
                # delete slime
                for i in range(len(slime)):
                    self.current_map[slime[i][1]][slime[i][0]] = self.classes.index('xxxx')
                continue
            # if close enough to do damage do not move, check every tile surrounding slime
            # slime has to be close to hew center of the screen
            if slime[0][0] > 56 and slime[0][0] < 61:
                is_player_close =  False
                is_player_close = is_player_close or self.current_map[slime[0][1]][slime[0][0]-1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[0][1]-1][slime[0][0]] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[0][1]-1][slime[0][0]-1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[1][1]][slime[1][0]+1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[1][1]-1][slime[1][0]] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[1][1]-1][slime[1][0]+1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[2][1]][slime[2][0]+1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[2][1]+1][slime[2][0]] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[2][1]+1][slime[2][0]+1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[3][1]][slime[3][0]-1] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[3][1]+1][slime[3][0]] == self.classes.index('player')
                is_player_close = is_player_close or self.current_map[slime[3][1]+1][slime[3][0]-1] == self.classes.index('player')
                if is_player_close:
                    continue

            if slime[0][0] < 58:
                tiles_right = self.classes[self.current_map[slime[1][1]][slime[1][0]+1]] == 'dirt'
                if not tiles_right:
                    up = 0
                    if self.classes[self.current_map[slime[2][1]][slime[2][0]+1]] == 'dirt':
                        up = -1
                    # moving slime to right, delet first and wirte new position
                    self.current_map[slime[0][1]][slime[0][0]] = self.classes.index('xxxx')
                    self.current_map[slime[1][1]][slime[1][0]] = self.classes.index('xxxx')
                    self.current_map[slime[2][1]][slime[2][0]] = self.classes.index('xxxx')
                    self.current_map[slime[3][1]][slime[3][0]] = self.classes.index('xxxx')

                    self.current_map[slime[0][1]+up][slime[0][0]+1] = self.classes.index('slime')
                    self.current_map[slime[1][1]+up][slime[1][0]+1] = self.classes.index('slime')
                    self.current_map[slime[2][1]+up][slime[2][0]+1] = self.classes.index('slime')
                    self.current_map[slime[3][1]+up][slime[3][0]+1] = self.classes.index('slime')
                    
                    slime = [ slime[1], (slime[1][0]+1, slime[1][1]), (slime[2][0]+1, slime[2][1]), slime[2]]
            else:
                tiles_left = self.classes[self.current_map[slime[0][1]][slime[0][0]-1]] == 'dirt'
                if not tiles_left:
                    # moving slime to left
                    up = 0
                    if self.classes[self.current_map[slime[3][1]][slime[3][0]-1]] == 'dirt':
                        up = -1
                    self.current_map[slime[0][1]][slime[0][0]] = self.classes.index('xxxx')
                    self.current_map[slime[1][1]][slime[1][0]] = self.classes.index('xxxx')
                    self.current_map[slime[2][1]][slime[2][0]] = self.classes.index('xxxx')
                    self.current_map[slime[3][1]][slime[3][0]] = self.classes.index('xxxx')

                    self.current_map[slime[0][1]+up][slime[0][0]-1] = self.classes.index('slime')
                    self.current_map[slime[1][1]+up][slime[1][0]-1] = self.classes.index('slime')
                    self.current_map[slime[2][1]+up][slime[2][0]-1] = self.classes.index('slime')
                    self.current_map[slime[3][1]+up][slime[3][0]-1] = self.classes.index('slime')

                    slime = [ (slime[0][0]-1, slime[0][1]), slime[0], slime[3], (slime[3][0]-1, slime[3][1])]
            # apply gravity
            is_grounded = self.classes[self.current_map[slime[2][1]+1][slime[2][0]]] == 'dirt' or self.classes[self.current_map[slime[3][1]+1][slime[3][0]]] == 'dirt'
            if not is_grounded:
                # moving down
                if slime[2][1]+1 < 67:
                    self.current_map[slime[2][1]+1][slime[2][0]] = self.classes.index('slime')
                    self.current_map[slime[3][1]+1][slime[3][0]] = self.classes.index('slime')

                self.current_map[slime[0][1]][slime[0][0]] = self.classes.index('xxxx')
                self.current_map[slime[1][1]][slime[1][0]] = self.classes.index('xxxx')

    def fixTrees(self, trees):
        # fix any floating tree left to player
        for j in trees:
            found_tree = False
            for i  in range(67):
                if self.classes[self.current_map[i][j]] == 'tree':
                    found_tree = True
                elif found_tree and self.classes[self.current_map[i][j]] == 'xxxx':
                    self.current_map[i][j] = self.classes.index('tree')
                elif self.classes[self.current_map[i][j]] == 'dirt' or self.classes[self.current_map[i][j]] == 'player' :
                    break

    def fixMap(self, jump=False):
        if not jump:
            # Check if player is in the air
            is_grounded = self.classes[self.current_map[34][58]] == 'dirt' or self.classes[self.current_map[34][59]] == 'dirt' 
            if not is_grounded:
                # fix player in the air
                new_map = np.full((67, 120), self.classes.index('xxxx'), dtype=np.int8)
                self.moveMap(new_map, new_i=-1)
                self.current_map = new_map.copy()
        
        # find all slimes and trees
        slimes = []
        trees = []
        positions = []
        for i in range(len(self.current_map)):
            for j in range(len(self.current_map[i])):
                if self.current_map[i][j] == self.classes.index('slime'):
                    if (j,i) in positions:
                        continue
                    slime_positions = self.findTiles(j, i, self.classes.index('slime'), [])
                    slimes.append(slime_positions)
                    positions = positions + slime_positions
                if self.current_map[i][j] == self.classes.index('tree'):
                    if j in trees:
                        continue
                    trees.append(j)

        # Move slime closer to player
        self.fixSlimes(slimes)

        self.fixTrees(trees)
        
    def apply_damage(self):
        close, _, _ = self.isEnemyOnAttackRange()
        if close:
            # enemy is close to player and will hit him, delete 3 heart 
            for j in range(len(self.current_map[0])-1, 0, -1):
                if self.classes[self.current_map[0][j]] == "heart":
                    self.current_map[0][j] = self.classes.index('xxxx')
                    self.current_map[0][j-1] = self.classes.index('xxxx')
                    self.current_map[0][j-2] = self.classes.index('xxxx')
                    break

    def findTiles(self, x, y, clss, positions = []):
        # delete recursively
        if x < 0 or y < 0 or y == len(self.current_map) or x == len(self.current_map[y]):
            return positions
        if self.current_map[y][x] != clss:
            return positions
        if (x,y) in positions:
            return positions
        
        positions.append((x,y))
        siblings = [(x+1,y), (x,y+1), (x+1,y+1), (x-1,y), (x,y-1), (x-1,y-1), (x+1, y-1), (x-1, y+1) ]
        for sibling in siblings:
            positions = self.findTiles(sibling[0], sibling[1], clss, positions=positions)
        return positions

    def getCloser(self):
        closest = 0
        min_distance = float('inf')
        rows = [8, 16, 24, 32, 40, 48, 56, 64]
        for i in rows:
            for j in range(len(self.current_map[i])):
                if self.current_map[i][j] == self.classes.index('tree'):
                    distance = abs(j - 58)
                    if distance < min_distance:
                        closest = j
                        min_distance = distance
        return closest

    def print(self):
        with open("delete.txt", 'w') as f:
            f.write(str(self.map))