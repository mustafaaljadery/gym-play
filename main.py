import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_map(reward_location):
    if reward_location == "bottom": 
        return ["SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"]
    
    return [
        "SFFFFFFG",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFF",
    ]

def run(episodes, is_training=True, render=False, reward_location = "bottom"):
    env = gym.make('FrozenLake-v1', map_name="8x8", desc=generate_map(reward_location), is_slippery=False, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        f = open('value.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 
    discount_factor_g = 0.9 
    epsilon = 1        
    epsilon_decay_rate = 0.0001        

    rng = np.random.default_rng()   
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  
        terminated = False      
        truncated = False     

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
            state = new_state
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    if is_training:
        f = open("value.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    # Training
    run(10000)

    # Deployment
    run(10, is_training=False, render=True, reward_location="bottom")

    # Misalignment
    run(10, is_training=False, render=True, reward_location="top")