
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


    