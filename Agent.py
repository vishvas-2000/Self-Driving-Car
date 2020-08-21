import random 
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.optimizers import Adam
import numpy as np 

class DQNAgent():
    def __init__(self, env, discount, learning_rate, exploration_decay, batch_size, memory_capacity):
        self.env = env
        self.discount = discount 
        self.lr = learning_rate
        self.batch_size = batch_size
        self.terminate = False
        self.num_actions = self.env.action_space.n
        self.obs_shape = np.array(env.observation_space.shape)
        
        self.epsilon = 1.0
        self.epsilon_decay = exploration_decay
        self.epsilon_start = 0.00
        self.experience = deque(maxlen= memory_capacity)

        self.model =  self.build_model()
        self.target_net = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(32, input_shape= self.obs_shape, activation = "relu"))
        model.add(layers.Dense(32, input_shape= self.obs_shape, activation = "relu"))
        model.add(layers.Dense(16, activation = "relu"))
        model.add(layers.Dense(self.num_actions, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(self.lr))
        return model
    
    def store(self, state, action, reward, next_state, terminated):
        self.experience.append((state, action, reward, next_state, terminated))
    
    def update_target_net(self):
        self.target_net.set_weights(self.model.get_weights())
        
    def get_exploration_rate(self, episode):
        return 0.001 + self.epsilon_start * np.power(self.epsilon_decay, episode)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        
        q_values = self.model.predict(tf.expand_dims(state, axis =0))
        return np.argmax(q_values[0])
    
    def train(self):
        minibatch = random.sample(self.experience, self.batch_size)
        
        X =[]
        y =[]
        
        for state, action, reward, next_state, terminated in minibatch:

            target = self.model.predict(tf.expand_dims(state, axis =0))
            
            if terminated:
                target[0][action] = reward 
            else:
                t = self.target_net.predict(tf.expand_dims(next_state, axis = 0))
                target[0][action] = reward + self.discount * np.amax(t)
                
            X.append(state)
            y.append(tf.squeeze(target))
            
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose = 0) 
        agent.model.save("C:/Models/Car-model")
        
        
agent = DQNAgent(env, discount= 0.99, learning_rate= 0.001, exploration_decay= 0.995, batch_size= 64, memory_capacity= 10_000)
