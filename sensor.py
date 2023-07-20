from GameObject import GameObject
from collisions import Collisions

class Sensor(GameObject):
    def __init__(self, x, y, search_angle, sensor_number):
        self.x = x 
        self.y = y 
        # passed in a sa  coordinate list ex. [1,1]
        self.search_angle = search_angle
        self.search_radius = 3
        self.max_search_distance = 20
        self.block_distance = 0
        self.active = True
        self.sensor_number = sensor_number

    def sense(self, collision_handler: Collisions, walls):
        if self.active:
            for distance in range(self.max_search_distance):
                block = GameObject(self.x + self.search_angle[0] * distance * self.search_radius, self.y + self.search_angle[1] * distance, self.search_radius, self.search_radius)
                for wall in walls:
                    if collision_handler.IsColliding(block,wall):
                        self.block_distance = distance
                        return self.block_distance
        else:
            return -1
                
        self.block_distance = self.max_search_distance
        return self.block_distance
    
    def Move_Sensor(self,x,y):
        self.x = x
        self.y = y

    def Deactivate(self):
        self.active = False

    def Activate(self):
        self.active = True