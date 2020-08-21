import pygame, sys
from pygame.math import Vector2
import numpy as np
from math import tan, sin, cos, radians, degrees, copysign

screen_width = 640
screen_height = 480

class race_car():
    def __init__(self):
        self.surface = pygame.image.load('car.png')
        self.position = Vector2(143.0, 401.0)
        self.vel = Vector2(35.0, 0.0)
        self.angle = -20.0
        self.speedlimit = 100
        self.max_acc = 20.0
        self.max_steer = 60.0
        self.len = 64 
        self.brake_deceleration = 100.0
        self.acc = 0.0
        self.steer = 0.0
        self.angular_vel = 0.0
        
    def radar(self, array,screen):
        radar = []
        for i in range(5):
            x1 = self.position.x 
            y1 = self.position.y
            angle = radians(self.angle + (i-2)*45)
            x = x1
            y = y1
            for j in range(300):
                x += cos(angle)
                y += sin(angle)
                if array[round(y),round(x),1]== 255:
                    break
            dist = np.sqrt((x-x1)**2 + (y-y1)**2)
            pygame.draw.circle(screen, (0,40+ 40*i,0),(int(x),int(y)),4)
            pygame.display.update()
            radar.append(int(dist))
        return radar
            
    def crash(self, array,screen):
        dist = self.radar(array, screen)
        for distance in dist:
            if distance < 12:
                return True
        return False
        
    def update(self, dt):
        self.vel.x += (self.acc * dt)
        self.vel.x = max(-self.speedlimit, min(self.vel.x, self.speedlimit))
        if self.steer:
            turning_radius = self.len / (3*sin(degrees(self.steer)))
            self.angular_vel = self.vel.x / turning_radius
        else:
            self.angular_vel = 0
        self.position += self.vel.rotate(self.angle) * dt
 
        self.angle += degrees(self.angular_vel) * dt
        if self.angle>360:
            self.angle = self.angle-360

    def draw(self, screen):
        rotated = pygame.transform.rotozoom(self.surface, -self.angle,1)
        rect = rotated.get_rect()
        screen.blit(rotated, self.position -(rect.width/2, rect.height/2))