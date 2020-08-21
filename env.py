import cv2  
import gym
from gym import spaces

class RaceCar2D():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.mode = 0
        self.car = race_car()
        self.maxSteps = 5_000
        self.steps = 0
        self.bg = pygame.image.load('map1.png') 
        self.track = cv2.imread('map1.png')
        self.track_array = np.array(self.track)
        self.dt = 1/15   
    
    def action(self, action):
        if action == 0:
            self.clock.tick(20)
            self.car.steer = 0
            self.car.acc = 0
            self.car.update(self.dt)
            self.steps += 1
            
        if action == 1:
            self.clock.tick(20)
            self.car.steer -= 600 * self.dt
            self.car.steer = max(-self.car.max_steer, min(self.car.steer, self.car.max_steer))
            self.car.acc = 0
            self.car.update(self.dt)
            self.steps += 1
            
        elif action == 2:
            self.clock.tick(20)
            self.car.steer += 600 * self.dt
            self.car.steer = max(-self.car.max_steer, min(self.car.steer, self.car.max_steer))
            self.car.acc = 0
            self.car.update(self.dt)
            self.steps += 1
        
    def evaluate(self):
        reward = 0 
        if not self.car.crash(self.track_array, self.screen):
            reward = 1
        else:
            reward = 0
        return reward
    
    def is_done(self):
        if self.car.crash(self.track_array, self.screen) or self.steps >= self.maxSteps:
            return True
        return False
    
    def observe(self):
        radar = list((np.array(self.car.radar(self.track_array, self.screen))-150)/50)
        return radar
    
    def view(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                done = True
                
        self.screen.blit(self.bg, (0, 0))
        self.car.draw(self.screen)
        
class CustomEnv():
    def __init__(self):
        self.pygame = RaceCar2D()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([-3, -3, -3, -3, -3]),np.array([3,3,3,3,3]), dtype=np.float32)

    def reset(self):
        del self.pygame
        self.pygame = RaceCar2D()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
        
env = CustomEnv()      