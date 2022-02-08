# -*- coding: utf-8 -*-

def mapper(obs):
    stat = 0
    if obs[0] > 0:
        stat += 1
    if obs[1] > 0:
        stat += 2
    if obs[2] > 0:
        stat += 4
    if obs[3] > 0:
        stat += 8
    if obs[4] > 0:
        stat += 16
    if obs[5] > 0:
        stat += 32
    return stat

import gym
import numpy as np

env = gym.make("Acrobot-v1")


Q = np.zeros((64, 3))

episodeS = 10
max_stepS = 500

learning_rate = 0.9
discount = 0.95
random_prob = 0.2

rewards = []
for episode in range(episodeS):
    observation = env.reset()
    state = mapper(observation)
    
    for _ in range(max_stepS):
        env.render()
        
        if np.random.uniform(0, 1) < random_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        next_observation, non_reward, done, _ = env.step(action)
        next_state = mapper(next_observation)
        reward = 1 - 0.5 * (next_observation[0] * (1 + next_observation[2]))
        
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount * np.max(Q[next_state, :]) - Q[state, action]) # This function updates the Q-table
        
        state = next_state
        
        if done:
            rewards.append(reward)
            random_prob -= 0.01
            break

env.close()
print(rewards)
