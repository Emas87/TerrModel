
class Inventory:
    def __init__(self):
        self.inventory = [[None for _ in range(10)] for _ in range(5)]
        self.ammo = [None for _ in range(4)]
        self.armor = [None for _ in range(3)]
        self.build = [None for _ in range(9)]
        self.inventory[0][0] = 'sword'
        self.inventory[0][1] = 'pickaxe'
        self.inventory[0][2] = 'axe'
    
    def updateInventory(self, row, column, object, count):
        self.inventory[row][column] = (object, count)
    
    def updateAmmo(self, index, object, count):
        self.ammo[index] = (object, count)
    
    def updateArmor(self, index, object, count):
        self.armor[index] = (object, count)
    
    def updateBuild(self, index, object, count):
        self.build[index] = (object, count)

    def getInventoryItemCount(self, object):
        for row in self.inventory:
            for item in self.inventory[row]:
                if item[0] == object:
                    return item[1]
                
    def getAmmoItemCount(self, object):
        for row in self.ammo:
            for item in self.ammo[row]:
                if item[0] == object:
                    return item[1]
                
    def getArmorItemCount(self, object):
        for row in self.armor:
            for item in self.armor[row]:
                if item[0] == object:
                    return item[1]
                
    def getBuildItemCount(self, object):
        for row in self.build:
            for item in self.build[row]:
                if item[0] == object:
                    return item[1]
    
    def __str__(self) -> str:
        str1 = 'Inventory:\n' + '\n'.join([' '.join([str(item) for item in row]) for row in self.inventory])
        str2 = '\nAmmo:\n' + ' '.join([str(item) for item in self.ammo])
        str3 = '\nArmor\n' + ' '.join([str(item) for item in self.armor])
        str4 = '\nBuild:\n' + ' '.join([str(item) for item in self.build]) + '\n'
        return str1 + str2 + str3 + str4


    