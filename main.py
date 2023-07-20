import random
import pygame
from pygame.locals import *
from car import Car
from sensor import Sensor
from wall import Wall
from checkpoint import checkpoint
from collisions import Collisions
import sys
import numpy as np
from tensorflow import keras
from keras import layers
import multiprocessing
import functools
import os
import json
"""
*********************************************

Think about using a different data structure for the walls
maybe a set or dict, then the sensor lookup should be faster
and easier to work with

*****************************************

"""
def main():

    pygame.display.set_caption("Drive IO")

    collisions = Collisions() 

    fps=24
    fpsclock=pygame.time.Clock()
    # 1600,1000
    screen = pygame.display.set_mode((1600,1000))
    car_Image = pygame.image.load('assets/car.png').convert()
    background = pygame.image.load('assets/background.jpg').convert()
    # screen.blit(background, (0, 0))
    White=(255,255,255)
    ticks_passed = 0
    max_ticks_per_generation = 10000

    # text
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    text_surface = my_font.render('Some Text', False, (0, 0, 0))

    sense_refresh_rate = 3

    # create n cars
    cars = []
    for _ in range(1):                    
        o = Car(10,10,15,15)
        cars.append(o)

    # Create walls
    # 200 pixels at top for text and analysis, 80x40
    walls = []
    checkpoints = []
    nums = ['0','1','2','3','4','5','6','7','8','9']
    letters = ['a','b','c','d','e','f','g']
    conversion_table = {
        "a": 11,
        "b": 13,
        "c": 13,
        "d": 14,
        "e": 15,
        "f": 16
    }
    with open("map1.txt") as map:
        lines = map.readlines()
        for y in range(len(lines)):
            for x in range(len(lines[y])):
                if lines[y][x] == "-":
                    pass
                elif lines[y][x] == "x":
                    wall = Wall(x,y)
                    walls.append(wall)
                elif lines[y][x] in nums:
                    zone =  checkpoint(x,y,int(lines[y][x]) + 1)
                    checkpoints.append(zone)
                elif lines[y][x] in letters:
                    number = conversion_table[lines[y][x]]
                    zone =  checkpoint(x,y,number)
                    checkpoints.append(zone)



    # Genetic Algorithm Parameters
    population_size = 32
    mutation_rate = 0.25
    num_generations = 20 

    # Create Initial Population
    def create_random_car():
        x = random.randint(10, 30)
        y = random.randint(10, 30)

        return Car(x, y, 15, 15) 

    current_generation = [create_random_car() for _ in range(population_size)]
    best_car = None

    # for car in current_generation: 
    #     print(car)
    #     print(type(car))

    # sys.exit()

    with open("best_cars_info.txt", "w") as f:
        f.write("Generation, Best Car Fitness, Num Cherckppoints Passed, Model Path\n")


    """
    Main Training loop
    """

    for generation in range(num_generations):

        for o in current_generation:
            o.x = 45 + random.randint(10, 30)
            o.y = 230 + random.randint(10, 30)
            o.fitness = 0

        simulate_generation(current_generation, max_ticks_per_generation, sense_refresh_rate, collisions, walls, checkpoints)


        current_generation.sort(key=lambda o: o.fitness, reverse=True)
        top_performers = current_generation[:population_size // 2]
        
        # Display the best-performing car from the last generation
        best_car = max(current_generation, key=lambda o: o.fitness)
        simulate_car(best_car, screen, walls, checkpoints, collisions, White, sense_refresh_rate, max_ticks_per_generation, fpsclock, fps)
        print("Generation:", generation + 1, "Best car fitness:", best_car.fitness)

        # save the data
        model_path = save_best_car_model(generation, best_car)
        with open("best_cars_info.txt", "a") as f:
            f.write(f"{generation + 1}, {best_car.fitness}, {best_car.checkpoints_passed}, {model_path}\n")

        offspring = []
        while len(offspring) < population_size - len(top_performers):
            parent1, parent2 = random.sample(top_performers, 2)
            if random.random() < mutation_rate:
                child_model = crossover(parent1.model, parent2.model, mutate=True)
            else:
                child_model = crossover(parent1.model, parent2.model)
            offspring.append(Car(10,10,15,15, child_model))

        current_generation = top_performers + offspring
        print("""
        GENERATION OVER
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        """)

def save_best_car_model(generation, best_car):
    folder_name = f"best_models/generation_{generation + 1}"
    os.makedirs(folder_name, exist_ok=True)
    model_path = os.path.join(folder_name, "best_model.h5")
    best_car.model.save(model_path)
    return model_path

# Function to create a child model by mixing weights from two parent models
# Function to clone a Keras model
def clone_keras_model(model):
    model_json = model.to_json()
    cloned_model = keras.models.model_from_json(model_json)
    cloned_model.set_weights(model.get_weights())
    return cloned_model

# Function to create a child model by mixing weights from two parent models
def crossover(parent1, parent2, mutate=False, mutation_rate=0.1):
    child_model = clone_keras_model(parent1)

    for layer_idx in range(len(parent1.layers)):
        # Get weights from parents
        parent1_weights = parent1.layers[layer_idx].get_weights()
        parent2_weights = parent2.layers[layer_idx].get_weights()

        # Mix the weights based on some mixing ratio (e.g., average)
        mixed_weights = []
        for w1, w2 in zip(parent1_weights, parent2_weights):
            mixed_weight = (w1 + w2) / 2  # You can try different mixing ratios
            mixed_weights.append(mixed_weight)

        # Assign the mixed weights to the child model
        child_model.layers[layer_idx].set_weights(mixed_weights)

    # Mutation (optional)
    if mutate:
        for layer in child_model.layers:
            weights = layer.get_weights()
            for i, w in enumerate(weights):
                if np.random.rand() < mutation_rate:
                    # Apply some small random changes to the weights
                    mutation = np.random.normal(loc=0, scale=0.1, size=w.shape)
                    weights[i] += mutation
            layer.set_weights(weights)

    return child_model


def simulate_car(car, screen, walls, checkpoints, collisions, White, sense_refresh_rate, max_ticks_per_generation, fpsclock, fps):
    ticks_passed = 0
    car.start()
    while True:
        screen.fill(White)
        for eve in pygame.event.get():
            if eve.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.draw.rect(screen, car.color, (car.x, car.y, car.width, car.height))
        image = pygame.image.load(car.image).convert()
        image = pygame.transform.rotate(image, car.image_rotate)
        screen.blit(image, (car.x - 15, car.y - 7))
        for w in walls:
            pygame.draw.rect(screen, w.color, (w.x, w.y, w.width, w.height))
        for c in checkpoints:
            pygame.draw.rect(screen, c.color, (c.x, c.y, c.width, c.height))

        car.update_with_neural_network()
        for w in walls:
            if collisions.IsColliding(car, w):
                car.stop()
        for c in checkpoints:
            if c.checkpoint_number - 1 == car.checkpoints_passed:
                if collisions.IsColliding(car, c):
                    car.Advance_Checkpoint()
        car.updateFitness(ticks_passed)
        if ticks_passed % sense_refresh_rate == 0:
            car.Use_Sensors(collisions, walls)
        if car.stopped or ticks_passed > max_ticks_per_generation:
            car.stop()
            break
        
        ticks_passed +=1
        pygame.display.update()
        fpsclock.tick(fps)

# Function to simulate a car and calculate its fitness without drawing anything
def simulate_car_no_draw(car, walls, collisions, checkpoints, sense_refresh_rate, max_ticks_per_generation):
    
    car.x = 45 + random.randint(10, 30)
    car.y = 230 + random.randint(10, 30)
    ticks_passed = 0
    car.start()
    while True:

        car.update_with_neural_network()
        for w in walls:
            if collisions.IsColliding(car, w):
                car.stop()
        for c in checkpoints:
            if c.checkpoint_number - 1 == car.checkpoints_passed:
                if collisions.IsColliding(car, c):
                    car.Advance_Checkpoint()
        car.updateFitness(ticks_passed)
        if ticks_passed % sense_refresh_rate == 0:
            car.Use_Sensors(collisions, walls)
        
        if car.stopped or ticks_passed > max_ticks_per_generation:
            break
        
        ticks_passed += 1
    
    return car

# Function to simulate cars in parallel for each generation
def simulate_generation(current_generation, max_ticks_per_generation, sense_refresh_rate, collisions, walls, checkpoints):
    # Use multiprocessing to simulate each car in parallel
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        partial_simulate_car_no_draw = functools.partial(
            simulate_car_no_draw,
            max_ticks_per_generation=max_ticks_per_generation,
            sense_refresh_rate=sense_refresh_rate,
            collisions=collisions,
            walls=walls,
            checkpoints=checkpoints
        )
        new_generation = pool.map(partial_simulate_car_no_draw, current_generation)

    # Update the current generation with the results from multiprocessing
    for i, car in enumerate(current_generation):
        current_generation[i] = new_generation[i]

if __name__ == "__main__":
    multiprocessing.freeze_support()
    pygame.init()
    pygame.font.init() 

    main()

# Calculate the best performer without drawing anything



# for generation in range(num_generations):
#     # Evaluation
#     for o in current_generation:
#         # Reset car position and fitness for the new evaluation
#         o.x = 45 + random.randint(10,30)
#         o.y = 230 + random.randint(10,30)
#         o.fitness = 0

#     # Simulation loop for each generation
#     for ticks_passed in range(max_ticks_per_generation):  # Define max_ticks_per_generation as needed
        
#         screen.fill(White)
#         for eve in pygame.event.get():
#             if eve.type==pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
        
#         # draw
#         for o in cars:
#             pygame.draw.rect(screen, o.color, (o.x, o.y, o.width , o.height))
#         for o in cars:
#             image = pygame.image.load(o.image).convert()
#             image = pygame.transform.rotate(image, o.image_rotate)
#             screen.blit(image, (o.x - 15, o.y- 7))
#         for w in walls:
#             pygame.draw.rect(screen, w.color, (w.x, w.y, w.width , w.height))
#         for c in checkpoints:
#             pygame.draw.rect(screen, c.color, (c.x, c.y, c.width , c.height))

#         # Simulation step
#         for o in current_generation:
#             # Replace this with your neural network or learning algorithm update
#             # based on sensor inputs and possibly car's previous actions
#             o.update_with_neural_network()  # Implement this method in the Car class

#             # Check for wall collisions
#             for w in walls:
#                 if collisions.IsColliding(o, w):
#                     o.stop()

#             # Update checkpoints and fitness
#             for c in checkpoints:
#                 if c.checkpoint_number - 1 == o.checkpoints_passed:
#                     if collisions.IsColliding(o, c):
#                         o.Advance_Checkpoint()

#             o.updateFitness(ticks_passed)

#             # Sense block distances
#             if ticks_passed % sense_refresh_rate == 0:
#                 o.Use_Sensors(collisions, walls)

#         # Check if all cars have reached the finish line
#         if all(o.checkpoints_passed == len(checkpoints) for o in current_generation) or all(o.stopped for o in current_generation):
#             break

#     # Selection
#     current_generation.sort(key=lambda o: o.fitness, reverse=True)
#     top_performers = current_generation[:population_size // 2]

#     # Crossover and Mutation
#     offspring = []
#     while len(offspring) < population_size - len(top_performers):
#         parent1, parent2 = random.sample(top_performers, 2)
#         if random.random() < mutation_rate:
#             child = crossover(parent1.model, parent2.model, mutate=True)
#         else:
#             child = crossover(parent1, parent2)
#         offspring.append(child)

#     # New Generation
#     current_generation = top_performers + offspring

# # Display the best-performing car from the last generation
# best_car = max(current_generation, key=lambda o: o.fitness)
# print("Best car fitness:", best_car.fitness)



# # game loop
# while 1:
#     screen.fill(White)
#     for eve in pygame.event.get():
#         if eve.type==pygame.QUIT:
#             pygame.quit()
#             sys.exit()
    
#     # draw
#     for o in cars:
#         pygame.draw.rect(screen, o.color, (o.x, o.y, o.width , o.height))
#     for o in cars:
#         image = pygame.image.load(o.image).convert()
#         image = pygame.transform.rotate(image, o.image_rotate)
#         screen.blit(image, (o.x - 15, o.y- 7))
#     for w in walls:
#         pygame.draw.rect(screen, w.color, (w.x, w.y, w.width , w.height))
#     for c in checkpoints:
#         pygame.draw.rect(screen, c.color, (c.x, c.y, c.width , c.height))


#     for o in cars:
#         # Input for testing base game, later will be NN
#         key_input = pygame.key.get_pressed()   
#         direction = [False,False,False,False]
#         if key_input[pygame.K_LEFT]:
#             direction[0] = True
#         if key_input[pygame.K_UP]:
#             direction[1] = True
#         if key_input[pygame.K_RIGHT]:
#             direction[2] = True
#         if key_input[pygame.K_DOWN]:
#             direction[3]=True

#         o.move(direction)

#         if key_input[pygame.K_SPACE]:
#             o.x = 55
#             o.y = 240 
#             o.start()
#             ticks_passed = 0
    
#     # check for wall collisions
#     for o in cars:
#         for w in walls:
#             if collisions.IsColliding(o,w):
#                 o.stop()
    
#     # update checkpoints and fitness
#     for o in cars:
#         for c in checkpoints:
#             if c.checkpoint_number - 1 == o.checkpoints_passed:
#                 if collisions.IsColliding(o,c):
#                     o.Advance_Checkpoint()

#         o.updateFitness(ticks_passed)
#         text_surface = my_font.render(f"{o.fitness}", False, (0, 0, 0))
#         screen.blit(text_surface, (0,0))

#     # sense block distances
#     if ticks_passed % sense_refresh_rate == 0:

#         for o in cars:
#             o.Use_Sensors(collisions,walls)
    
#     ticks_passed +=1


#     pygame.display.update()
#     fpsclock.tick(fps)



