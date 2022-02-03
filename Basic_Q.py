# -*- coding: utf-8 -*-

import gym
import numpy as np
import time

env = gym.make("FrozenLake-v0")
stateS = env.observation_space.n
actionS = env.action_space.n

Q = np.zeros((stateS, actionS))

episodeS = 1500
max_stepS = 100

learning_rate = 0.81
gamma = 0.96

epsilon = 0.9

rewards = []
for episode in range(episodeS):
    
    state = env.reset()
    for _ in range(max_stepS):
        
        env.render()
        
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        next_state, reward, done, _ = env.step(action)
        
        Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state
        
        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break

print(Q)
print(f"average reward: {sum(rewards)/len(rewards)}:")