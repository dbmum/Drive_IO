import pygame
import sys
pygame.init()
fps=30
fpsclock=pygame.time.Clock()
sur_obj=pygame.display.set_mode((1600,1000))
pygame.display.set_caption("Drive IO")
White=(255,255,255)
x=10
y=10
step=2
left_speed = 0
up_speed = 1
right_speed = 2
down_speed = 3
velocity = [0,0,0,0]
friction = .5
max_speed = 10
player = pygame.image.load('assets/car.png').convert()

while True:
    sur_obj.fill(White)
    pygame.draw.rect(sur_obj, (255,0,0), (x, y, 20, 20))
    # sur_obj.blit()
    for eve in pygame.event.get():
        if eve.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
    key_input = pygame.key.get_pressed()   
    if key_input[pygame.K_LEFT]:
        velocity[left_speed] -= step
    if key_input[pygame.K_UP]:
        velocity[up_speed] -= step
    if key_input[pygame.K_RIGHT]:
        velocity[right_speed] += step
    if key_input[pygame.K_DOWN]:
        velocity[down_speed] +=step
    
    for i in range(len(velocity)):
        if velocity[i] > 0:
            if velocity[i] > max_speed:
                velocity[i] = max_speed
            else:
                velocity[i] -= friction

        elif velocity[i] < 0:
            if velocity[i] < -max_speed:
                velocity[i] = -max_speed
            else:
                velocity[i]+=friction
            
    
    x += (velocity[left_speed] + velocity[right_speed])
    y += (velocity[down_speed] + velocity[up_speed])
    pygame.display.update()
    fpsclock.tick(fps)