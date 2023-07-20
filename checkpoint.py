from GameObject import GameObject

class checkpoint(GameObject):
    def __init__(self, x, y, checkpoint_number):
        self.x = x * 20 
        self.y = y * 20 + 200
        self.width = 20
        self.height = 20
        self.color = (0,0,255)
        self.checkpoint_number = checkpoint_number
