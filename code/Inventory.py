
import numpy as np
class Inventory:
    def __init__(self, classes):
        # first 5 rows are for inventory
        # row 6 for ammo
        # row 7 for armor
        # row 8 for build
        self.classes = classes
        #self.inventory = [[self.classes.index('x') for _ in range(10)] for _ in range(9)]
        self.inventory = np.full((9, 10), self.classes.index('xxxx'), dtype=np.int8)

        #self.ammo = [None for _ in range(4)]
        #self.armor = [None for _ in range(3)]
        #self.build = [None for _ in range(9)]
        if 'sword' in self.classes:
            self.inventory[0][0] = self.classes.index('sword')
        if 'pickaxe' in self.classes:
            self.inventory[0][1] = self.classes.index('pickaxe')
        if 'axe' in self.classes:
            self.inventory[0][2] = self.classes.index('axe')
    
    def updateInventory(self, row, column, object):
        self.inventory[row][column] = self.classes.index(object)
    
    def updateAmmo(self, index, object):
        self.inventory[5][index] = self.classes.index(object)
    
    def updateArmor(self, index, object):
        self.inventory[6][index] = self.classes.index(object)
    
    def updateBuild(self, index, object):
        self.inventory[7][index] = self.classes.index(object)
    
    def __str__(self) -> str:
        str1 = 'Inventory:\n' + '\n'.join([' '.join([self.classes[item][:4] for item in row]) for row in self.inventory]) + '\n'
        return str1

    def getWood(self):
        for i in range(len(self.inventory)):
            for j in range(len(self.inventory[i])):
                if self.inventory[i][j] == self.classes.index('wood'):
                    return True, j, i
        return False, 0, 0
    
    def getBuild(self, item):
        for j in range(len(self.inventory[7])):
            if self.inventory[7][j] == self.classes.index(item):
                return True, j
        return False, 0
    
    def convertCoords(self, col, row):        
        inventory_min = (20, 20)
        inventory_max = (20 + int(9*52.5) + 50, 20 + int(4*52.5) + 50)
        slot_size = (int((inventory_max[0]-inventory_min[0])/10), int((inventory_max[1]-inventory_min[1])/5))
        center_diff = (col*slot_size[0], row*slot_size[1])
        center = (center_diff[0] + inventory_min[0], center_diff[1] + inventory_min[1] )
        #x = int(center[0] - slot_size[0]/2)
        #y = int(center[1] - slot_size[1]/2)
        x = center[0]
        y = center[1]
        return x, y, slot_size[0], slot_size[1]
        # result center[0] - slot_size[0]/2, center[1], slot_size[0], slot_size[1]/2
        center_diff = (center[0]-inventory_min[0], center[1]-inventory_min[1])
        slot_size = ((inventory_max[0]-inventory_min[0])/10, (inventory_max[1]-inventory_min[1])/5)
        col = floor(center_diff[0]/slot_size[0])
        row = floor(center_diff[1]/slot_size[1])

    def convertCoordsBuild(self, row):
        
        build_min = (24, 420)
        build_max = (24 + 54, 690 + int(4*54))    

        slot_size = (build_max[1]-build_min[1])/9
        center_diff = slot_size*row
        center = center_diff + build_min[1]
        x = int(30)
        y = int(center - slot_size/2)
        w = slot_size
        h = slot_size
        return x, y, w, h
        # result center[0] - slot_size[0]/2, center[1], slot_size[0], slot_size[1]/2

    