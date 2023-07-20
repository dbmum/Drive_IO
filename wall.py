from GameObject import GameObject

class Wall(GameObject):
    def __init__(self, x, y):
        self.x = x * 20 
        self.y = y * 20 + 200
        self.width = 20
        self.height = 20
        self.color = (0,255,0)