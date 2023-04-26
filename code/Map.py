
from math import floor
from time import time
class Map:
    def __init__(self) -> None:
        self.current_map = self.inventory = [[(None, 0) for j in range(120)] for i in range(67)]
        self.max = (1920, 1080)
        self.min = (0,0)
        self.max_tiles = (120, 67)

    def updateMap(self, x1, y1, w, h, object):
        slot_size = (16, 16)
        for x_tile in range(0, int(w/slot_size[0]) + 1):
            for y_tile in range(0, int(h/slot_size[1]) + 1):
                corner  = (x1 + slot_size[0]*x_tile, y1 + slot_size[1]*y_tile)
                corner_diff = (corner[0]-self.min[0], corner[1]-self.min[1])
                row = floor(corner_diff[1]/slot_size[1])
                col = floor(corner_diff[0]/slot_size[0])
                tile_time = time()
                self.current_map[row][col] = (object, tile_time)

    def __str__(self) -> str:
        str1 = 'Map:\n' + '\n'.join([' '.join([str(item) for item in row]) for row in self.current_map])
        return str1