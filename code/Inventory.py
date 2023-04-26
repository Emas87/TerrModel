
class Inventory:
    def __init__(self):
        self.inventory = [[None for j in range(10)] for i in range(5)]
        self.ammo = [None for i in range(4)]
        self.armor = [None for i in range(3)]
        self.build = [None for i in range(9)]
        self.inventory[0][0] = 'sword'
        self.inventory[0][1] = 'pickaxe'
        self.inventory[0][2] = 'axe'
    
    def updateInventory(self, row, column, object):
        self.inventory[row][column] = object
    
    def updateAmmo(self, index, object):
        self.ammo[index] = object
    
    def updateArmor(self, index, object):
        self.armor[index] = object
    
    def updateBuild(self, index, object):
        self.build[index] = object
    
    def __str__(self) -> str:
        str1 = 'Inventory:\n' + '\n'.join([' '.join([str(item) for item in row]) for row in self.inventory])
        str2 = '\nAmmo:\n' + ' '.join([str(item) for item in self.ammo])
        str3 = '\nArmor\n' + ' '.join([str(item) for item in self.armor])
        str4 = '\nBuild:\n' + ' '.join([str(item) for item in self.build]) + '\n'
        return str1 + str2 + str3 + str4


    