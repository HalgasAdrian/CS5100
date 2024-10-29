import time
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial 
	   configuration and taking actions until a terminal state is reached.
	2. Keep track of gameplay history in an appropriate format for each of the episodes.
	3. From gameplay history, estimate the probability of victory against each of the guards when taking the fight action.

	Some important notes:

		a. Keep in mind that given some observation [(X,Y), health, guard_in_cell], a fight action is only meaningful if the 
		   last entry corresponding to guard_in_cell is nonzero.

		b. Upon taking the fight action, if the player defeats the guard, the player is moved to a random neighboring cell with 
		   UNCHANGED health. (2 = Full, 1 = Injured, 0 = Critical).

		c. If the player loses the fight, the player is still moved to a random neighboring cell, but the health decreases by 1.

		d. Your player might encounter the same guard in different cells in different episodes.

		e. All interaction with the environment must be done using the env.step() method, which returns the next
		   observation, reward, done (Bool indicating whether terminal state reached) and info. This method should be called as 
		   obs, reward, done, info = env.step(action), where action is an integer representing the action to be taken.

		f. The env.reset() method resets the environment to the initial configuration and returns the initial observation. 
		   Do not forget to also update obs with the initial configuration returned by env.reset().

		g. To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		   For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		   will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		h. To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		   Example usage below. This function should be called after every action.

		   if gui_flag:
		       refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the np array, P which contains four float values, each representing the probability of defeating guards 1-4 respectively.

'''

def estimate_victory_probability(num_episodes=70000):
    guard_encounters = np.zeros(4)
    guard_victories = np.zeros(4)

    fight_action_index = 4  # 'FIGHT' action is numerically indexed as 4

    for _ in range(num_episodes):
        result = env.reset()
        obs = result if isinstance(result, dict) else result[0]
        done = False

        while not done:
            # Bias the action selection towards fighting if a guard is present in the cell
            if obs.get('guard_in_cell'):
                action = fight_action_index
            else:
                action = env.action_space.sample()  # Otherwise, sample a random action

            result = env.step(action)
            next_obs, reward, done, info = result if isinstance(result, tuple) else (result, 0, True, {})

            if gui_flag:
                refresh(obs, reward, done, info)  # GUI refresh

            print(f"Action taken: {action}, Guard in cell: {obs.get('guard_in_cell', 'None')}, Reward: {reward}")

            if action == fight_action_index and obs.get('guard_in_cell'):
                guard_id = int(obs['guard_in_cell'][1]) - 1
                guard_encounters[guard_id] += 1

                if reward > 0:
                    guard_victories[guard_id] += 1

            obs = next_obs  # Update obs for the next iteration

    P = guard_victories / np.maximum(guard_encounters, 1)
    return P

probabilities = estimate_victory_probability(70000)
print("Estimated probabilities of defeating each guard:", probabilities)


