# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:57:52 2017

@author: momos_000
"""
# using tabular Q-Learning to learn a flat policy in RoomWorld
# referring to U.C. Berkeley DeepRL Bootcamp materials

import time
import numpy as np
from room_world import RoomWorld, Agent_Q, Agent_Q_ER
import learning_test_utilities as util
import random


env          = RoomWorld()
state_space  = env.state_space
num_actions  = env.action_space.size
q_func       = util.QTable_Numpy(env.walkability_map.shape,num_actions) #according to single test, Numpy version is 2.7x faster
agent_q      = Agent_Q_ER(env,q_func)
cur_state    = env.reset(random_placement=True)
#training
max_steps  = 1000
iterations, epsilon, gamma, alpha = util.learning_parameters()
report_freq = iterations/10
hist = np.zeros((iterations,7)) #training step, avg_td, avg_ret, avg_greedy_ret, avg_greedy_successrate, avg_greedy_steps, avg_greedy_choices
start_time = time.time()
batch_size = 32


def replay(batch_size):
    minibatch = random.sample(agent_q.memory, batch_size)
    tot_td = 0.0
    stp = 0
    for state, action, reward, next_state, done in minibatch:
        tde     = util.q_learning_update(gamma, alpha, agent_q.q_func, state, action, next_state, reward)
        tot_td += tde
        stp += 1
    return tot_td, stp

for itr in range(iterations):
    tot_td = 0
    cur_state = env.reset(random_placement=True)
    stp = 0
    rewards = []
    done = False
    while not done and stp<max_steps:
        # env.render()
        action  = agent_q.epsilon_greedy_action(cur_state,eps=epsilon)
        next_state, reward, done = env.step(action)
        agent_q.remember(cur_state, action, reward, next_state, done)
        rewards.append(reward)
        stp += 1
        #replay for agent
        if batch_size < len(agent_q.memory):
            err, _ = replay(batch_size)
            tot_td += err
            # stp += replay_stp
        
        cur_state = next_state
        
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(itr, iterations, stp, epsilon))
            break
  
    # record results for this iteration
    prev_steps = hist[itr-1,0]
    greedy_steps, greedy_choices, greedy_ret, greedy_success = util.greedy_eval(agent_q,gamma,max_steps,100)
    hist[itr,:] = np.array([prev_steps+stp, tot_td/(stp), util.discounted_return(rewards,gamma), greedy_ret, greedy_success, greedy_steps, greedy_choices])
    
    if itr % report_freq == 0: # evaluation
        print("Itr %i # Average reward: %.2f" % (itr, hist[itr,3]))

print("DONE. ({} seconds elapsed)".format(time.time()-start_time)) 
util.plot_and_pickle(env,agent_q,hist)
