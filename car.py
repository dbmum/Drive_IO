from GameObject import GameObject
from sensor import Sensor
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Function to threshold actions based on predicted probabilities
def _threshold_actions(predictions, threshold=0.5, max_actions=2):
    # Sort the probabilities and get indices in descending order
    sorted_indices = tf.argsort(predictions, direction='DESCENDING')

    # Initialize the binary vector indicating selected actions
    selected_actions = tf.zeros_like(predictions, dtype=tf.bool)

    # Select the top max_actions actions above the threshold
    num_selected = 0
    for idx in sorted_indices:
        if predictions[idx] >= threshold and num_selected < max_actions:
            selected_actions = tf.tensor_scatter_nd_update(selected_actions, [[idx]], [True])
            num_selected += 1

    return selected_actions

class Car(GameObject):
    def __init__(self, x, y, width, height, model = None):
        super().__init__(x, y, width, height)
        self.x = x + 45
        self.y = y + 230
        self.step=1.5
        self.left_speed = 0
        self.up_speed = 1
        self.right_speed = 2
        self.down_speed = 3
        self.velocity = [0,0,0,0]
        self.friction = .5
        self.max_speed = 8
        self.color = (255,0,0)
        self.image = "assets/car.png"
        self.image_rotate = 0
        self.image_x_offset = 0
        self.image_y_offest = 0
        self.checkpoints_passed = 0
        self.fitness = 0
        self.stopped = False
        self.blocks_sensed_distance = [0,0,0,0,0,0,0,0]

        if model is not None:
            self.model = model 
        else: 
            input_size = len(self.blocks_sensed_distance)

            # Define the Keras model
            self.model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_size,)),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(4, activation='sigmoid')  
            ])

            # Compile the model
            self.model.compile(optimizer='adam', loss='mse')  


        self.sensors = []

        sensor_angles = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]

        # Sensor 0 is the 12 o'clock position, then it goes up clockwise at 45 degree angle increments
        for i in range(len(sensor_angles)):
            sensor = Sensor(self.x + (self.width / 2), self.y + (self.height / 2), sensor_angles[i], i)
            self.sensors.append(sensor)

            
    

    def move(self, direction):
        if direction[0]:
            self.velocity[self.left_speed] -= self.step
            self.image_rotate = 180
        if direction[1]:
            self.velocity[self.up_speed] -= self.step
            self.image_rotate = 90
        if direction[2]:
            self.velocity[self.right_speed] += self.step
            self.image_rotate = 0
        if direction[3]:
            self.velocity[self.down_speed] += self.step
            self.image_rotate = 270

        # angled image rotation
        if direction[0] and direction[1]:
            self.image_rotate = 135
        if direction[1] and direction[2]:
            self.image_rotate = 45
        if direction[2] and direction[3]:
            self.image_rotate = 315
        if direction[3] and direction[0]:
            self.image_rotate = 225
        
        
        
        
        for i in range(len(self.velocity)):
            if self.velocity[i] > 0:
                if self.velocity[i] > self.max_speed:
                    self.velocity[i] = self.max_speed
                else:
                    self.velocity[i] -= self.friction

            elif self.velocity[i] < 0:
                if self.velocity[i] < -self.max_speed:
                    self.velocity[i] = -self.max_speed
                else:
                    self.velocity[i]+=self.friction
                
        
        self.x += (self.velocity[self.left_speed] + self.velocity[self.right_speed])
        self.y += (self.velocity[self.down_speed] + self.velocity[self.up_speed])

        # move sensors with the car
        for sensor in self.sensors:
            
            sensor.Move_Sensor(self.x + (self.width / 2), self.y + (self.height / 2))
        
        # deactivate unused sensors
        # Refer to car_rotation_diagram_with_sensor_numbers for these numbers
        activate = []
        if self.image_rotate ==  0:
            activate = [1,2,3]
        elif self.image_rotate ==  315:
            activate = [2,3,4]
        elif self.image_rotate ==  270:
            activate = [3,4,5]
        elif self.image_rotate ==  225:
            activate = [4,5,6]
        elif self.image_rotate ==  180:
            activate = [5,6,7]
        elif self.image_rotate ==  135:
            activate = [6,7,0]
        elif self.image_rotate ==  90:
            activate = [7,0,1]
        elif self.image_rotate ==  45:
            activate = [0,1,2]
        
        for sensor in self.sensors:
            if sensor.sensor_number in activate:
                sensor.Activate()
            else:
                sensor.Deactivate()
        

    # @tf.function
    # def _threshold_actions(self, predictions, threshold=0.5, max_actions=2):
    #     # Sort the probabilities and get indices in descending order
    #     sorted_indices = tf.argsort(predictions, direction='DESCENDING')

    #     # Initialize the binary vector indicating selected actions
    #     selected_actions = tf.zeros_like(predictions, dtype=tf.bool)

    #     # Select the top max_actions actions above the threshold
    #     num_selected = 0
    #     for idx in sorted_indices:
    #         if predictions[idx] >= threshold and num_selected < max_actions:
    #             selected_actions = tf.tensor_scatter_nd_update(selected_actions, [[idx]], [True])
    #             num_selected += 1

    #     return selected_actions

    def update_with_neural_network(self):
        predicted_probs = self.model.predict(np.expand_dims(self.blocks_sensed_distance, axis=0))[0]

        threshold = 0.25
        max_actions = 2

        selected_actions = _threshold_actions(predicted_probs, threshold=threshold, max_actions=max_actions)
        
        self.move(selected_actions)
        # print(selected_actions)

        
    
    def stop(self):
        self.max_speed = 0
        self.velocity = [0,0,0,0]
        self.image = "assets/blue_car.png"
        self.stopped = True
        if self.checkpoints_passed == 0:
            self.fitness = self.x - 100
    
    def start(self):
        self.max_speed = 8
        self.image = "assets/car.png"
        self.stopped = False
        self.fitness = 0
        self.checkpoints_passed = 0
    
    def updateFitness(self, tick):
        if self.stopped:
            return
        self.fitness = round((30 - tick /100) * (self.checkpoints_passed + 1), 3)
    
    def Advance_Checkpoint(self):
        self.checkpoints_passed +=1

    def Use_Sensors(self, collisions, walls):
        new_sense = [0,0,0,0,0,0,0,0]
        for i in range(len(self.sensors)):
            
            new_sense[i] = self.sensors[i].sense(collisions, walls)
            self.blocks_sensed_distance = new_sense
        # print(self.blocks_sensed_distance)
        