# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:24:18 2023

FACTORISED AFFORDANCE MODEL
EXPERIMENT: TOOL INNOVATION

@authors: pfkin + pzc
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'pymdp-sparse_likelihoods_111' ))
import pymdp
from pymdp import utils
from tool_making_environment import toolEnv as Environ
from plots import * # plotting functions
from agent import Agent # this is a modified Agent class from original pymdp script
import control # this is a modified control script from original pymdp  
import numpy as np
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def fill_A(A,prob=1., reward_location=2):
    """
    Parameters
    ----------
    A : pymdp obj_array_zeros
        empty observation likelihood array or `A`
    prob : float
        The default is 1.
    reward_location : integer
        room index where reward is located

    Returns
    -------
    A : A matrix
        Filled observation likelihood array (A matrix) 
    """
    # 4 arrays, 1 for each observation dimension of size: 
    # obs[i].size * state[0].size * state[1].size * state[2].size
    for i in range(A[0].shape[3]): # for each room
        non_prob = (1-prob)/(A[0].shape[0]-1)
        A[0][:,:,:,i] = np.ones((A[0].shape[0], A[0].shape[1],A[0].shape[2]))*non_prob
        A[0][0,0,0,i] = prob 
        A[0][2,1,0,i] = prob  
        A[0][1,0,1,i] = prob
        A[0][3,1,1,i] = prob
    
    for i in range(A[1].shape[1]):
        for ii in range(A[1].shape[2]):
            non_prob = (1-prob)/(A[1].shape[0]-1)
            A[1][:,i,ii,:] = np.ones((A[1].shape[0], A[1].shape[3]))*non_prob
            np.fill_diagonal(A[1][:,i,ii,:], prob)

    # reward_location relates to particular [x_reach, y_reach. room]
    # note: order of tool HV or VH does not matter
    lookup = {2:[1,1,1], 3:[1,0,1], 4:[1,0,0], 5:[1,1,0] , 6:[0,1,0], 7:[0,1,1]}  
    reward_x_reach = lookup[reward_location][0]
    reward_y_reach = lookup[reward_location][1]
    reward_room = lookup[reward_location][2]
    
    # tell agent what room and tool states combinations will get reward
    non_prob = (1-prob)/(A[2].shape[0]-1)
    A[2][0,:,:,:] = prob 
    A[2][1,:,:,:] = non_prob
    A[2][0,reward_x_reach,reward_y_reach,reward_room] = non_prob 
    A[2][1,reward_x_reach,reward_y_reach,reward_room] = prob
    
    return None
    

def build_B_room(B, fully_known,  prob):
    """
    Parameters
    ----------
    B : pymdp obj_array 
        random transition likelihood (B matrix)
    fully_known : bool
        indicates whether matrix given a priori or must be learned
    prob : float
        The default is 1.

    Returns
    -------
    B : constructs B[2] of B matrix
        Room state depends on current room (and action)

    """
    if fully_known:
        non_prob = (1-prob)/(B[2].shape[0]-1)
        # room state, null action
        B[2][:,:,0] = np.ones((B[2].shape[0], B[2].shape[1]))*non_prob
        np.fill_diagonal(B[2][:,:,0], prob)
        # room state, move action
        B[2][:,:,1] = np.ones((B[2].shape[0], B[2].shape[1]))*non_prob
        B[2][0,1,1] = prob
        B[2][1,0,1] = prob
        # room state, pick-up action
        B[2][:,:,2] = np.ones((B[2].shape[0], B[2].shape[1]))*non_prob
        np.fill_diagonal(B[2][:,:,2], prob)
        # room state, drop-off action
        B[2][:,:,3] = np.ones((B[2].shape[0], B[2].shape[1]))*non_prob
        np.fill_diagonal(B[2][:,:,3], prob)
        
    else:
        # each B[:,:,i] is the same everywhere adding up to 1.0 for B[2]
        for i in range(B[2].shape[2]):
            B[2][:,:,i] = np.ones((B[2].shape[0],B[2].shape[1])) / B[2][:,:,i].shape[0]   

    return B


def build_B_x_reach(B, fully_known, prob):
    """
    Parameters
    ----------
    B : pymdp obj_array 
        random transition likelihood (B matrix)
    fully_known : bool
        indicates whether matrix given a priori or must be learned
    prob : float
        The default is 1.

    Returns
    -------
    B : constructs B[0] of B matrix
        x_reach depends on current x_reach and room (and action)
    
    """
    if fully_known:
        non_prob = (1-prob)/(B[0].shape[0]-1)
                
        # action = null:
        # room = 0
        B[0][:,:,0,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,0], prob) # = identity matrix
        # room = 1
        B[0][:,:,1,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,0], prob)# = identity matrix
        
        # action = move:
        # room = 0 
        B[0][:,:,0,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,1], prob) # = identity matrix
        # room = 1
        B[0][:,:,1,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,1], prob) # = identity matrix
        
        # action = pickup
        # room = 0
        B[0][:,:,0,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,2], prob) # = identity matrix
        # room = 1
        B[0][:,:,1,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        #np.fill_diagonal(B[0][:,:,1,0], prob)# 
        B[0][:,:,1,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,1,1,2] = prob
        B[0][1,0,1,2] = prob # = opposite of identity matrix
        
        # action = drop
        # room = 0
        B[0][:,:,0,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,0,3] = prob # = block top row
        # room = 1
        B[0][:,:,1,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,1,3] = prob # = block top row
        
    else:
        for i in range(B[0].shape[2]):
            for j in range(B[0].shape[3]):
                B[0][:,:,i,j] = np.ones((B[0].shape[0],B[0].shape[1])) / B[0][:,:,i,j].shape[0]   
        
    return B


def build_B_y_reach(B, fully_known, prob):
    """
    Parameters
    ----------
    B : pymdp obj_array 
        random transition likelihood (B matrix)
    fully_known : bool
        indicates whether matrix given a priori or must be learned
    prob : float
        The default is 1.

    Returns
    -------
    B : constructs B[1] of B matrix
        y_reach depends on current y_reach and room (and action)
    """
    if fully_known:
        non_prob = (1-prob)/(B[1].shape[0]-1)
        
        # action = null:
        # room = 0
        B[1][:,:,0,0] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,0,0], prob) # = identity matrix
        # room = 1
        B[1][:,:,1,0] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,1,0], prob) # = identity matrix
        
        # action = move:
        # room = 0 
        B[1][:,:,0,1] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,0,1], prob) # = identity matrix
        # room = 1
        B[1][:,:,1,1] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,1,1], prob) # = identity matrix
        
        # action = pickup
        # room = 0
        B[1][:,:,0,2] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][:,:,0,2] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][0,1,0,2] = prob
        B[1][1,0,0,2] = prob # = opposite of identity matrix
        # room = 1
        B[1][:,:,1,2] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob 
        np.fill_diagonal(B[1][:,:,1,2], prob) # = identity matrix
        
        # action = drop
        # room = 0
        B[1][:,:,0,3] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][0,:,0,3] = prob # = block top row
        # room = 1
        B[1][:,:,1,3] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][0,:,1,3] = prob # = block top row
    
        
    else:
        for i in range(B[1].shape[2]):
            for j in range(B[1].shape[3]):
                B[1][:,:,i,j] = np.ones((B[1].shape[0],B[1].shape[1])) / B[1][:,:,i,j].shape[0] 
        
    return B
    
def fill_B(B, prob=1., fully_known=False):
    """
    Parameters
    ----------
    B : pymdp obj_array 
        random transition likelihood (B matrix)
    prob : float
        The default is 1.
    fully_known : bool
        default is False: indicates whether matrix filled in a priori
    
    Returns
    -------
    B : B matrix
        Filled transition likelihood array 

    """
    build_B_x_reach(B, fully_known, prob)
    build_B_y_reach(B, fully_known, prob)
    build_B_room(B, fully_known, prob)
    
    return None
    

def fill_C(C, punish, reward):
    """
    Parameters
    ----------
    C : pymdp obj_array_uniform 
        uniform prior preference array (C vector)
    punish : integer
        relative log probability of observing 0 in reward modality
    reward : integer
        relative log probability of observing 1 in reward modality
    Returns
    -------
    C : C vector
        Filled prior preference over observation array 

    """
    C[2][0] = punish
    C[2][1] = reward
    
    return None

# _____________________________________________________________________________

#                       CONSTRUCT GENERATIVE MODEL

#______________________________________________________________________________

# observations
obs_room = ["Left_room", "Right_room"]
obs_tool = ["None", "V", "H", "HV"]
obs_reward = ["Punish", "Reward"]   

# states
state_room = ["Left_room", "Right_room"]
state_x_reach = ["x_Reach 0", "x_Reach 1"]
state_y_reach = ["y_Reach 0", "y_Reach 1"]

# controls
room_action = ["Null", "Move", "Pick-up", "Drop"]
x_action = ["Null", "Move", "Pick-up", "Drop"]
y_action = ["Null", "Move", "Pick-up", "Drop"]
combined_action_list=[room_action, x_action, y_action]

 # observation modality dimensions
num_obs = [len(obs_tool), len(obs_room),len(obs_reward)]
# hidden state factor dimensions
num_states = [len(state_x_reach),len(state_y_reach),len(state_room)] 
# control state factor dimensions
num_controls = [len(x_action),len(y_action),len(room_action)]

# set dummy reward location to allow us to initialise the agent  
reward_location = 3

# create observation likelihood (A matrix)
A = utils.initialize_empty_A(num_obs, num_states) 
prob_A = 1. # confidence in A matrix
fill_A(A, prob=prob_A, reward_location=reward_location)
dir_scale = 1.
pA=utils.dirichlet_like(A,scale=dir_scale)

# B_factor_list says the states each state depends on
# 0 (x_reach), 1 (y_reach), 2 (room state)
B_factor_list = [[0,2],[1,2],[2]]

# create transition likelihood (B matrix) with factor_list
B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list) 
prob_B = 1. # confidence in B matrix
fill_B(B, fully_known = False, prob=prob_B)
pB = utils.dirichlet_like(B,scale=dir_scale)

# create prior preference over observations (C vector)
punish, reward = 0, 20
C = utils.obj_array_uniform(num_obs)
fill_C(C, punish, reward)

# instantiate agent
policy_len = 4
policies = None
policies_restriction= "single_action"
# before restricting policies to just a sngle aciton
# need to ensure that all the action lists are the same
assert all(x==combined_action_list[0] for x in combined_action_list), "For a single action, you must set up actions for each hidden state to be identical"
my_agent = Agent(A=A, pA=pA, B = B, pB=pB, C = C, policy_len=policy_len, policies_restriction=policies_restriction, policies=policies,B_factor_list=B_factor_list, action_selection="stochastic")

# environment
init_room = 0 
init_tool = 0
env = Environ(reduced_obs=False, init_state=init_room, init_tool=init_tool, reward_location=reward_location)
locations =  [6,6,6,6,3,3,3,3,4,4,4,4,7,7,7,7,2,2,2,2,5,5,5,5] # changing reward locations
num_runs = len(locations)
steps_per_run = 10
# _____________________________________________________________________________

#                                 EXPERIMENT

#______________________________________________________________________________

def run_experiment(locations, num_runs, steps_per_run, num_obs, num_states, prob_A, my_agent, env, init_room, init_tool):
    
    # track stats for plotting functions
    num_steps_history = []
    tool_certainty = []
    
    utility_rank_history = []
    infogain_rank_history = []
    G_rank_history = []

    
    filecount = 0
    
    # loop over runs
    for i in range(num_runs):
    
        # select reward location for this run  
        reward_location = locations[i]
        # A matrix recreated at start of each run since always fully known
        # reward location changes throughout run which is why this is necessary
        A = utils.initialize_empty_A(num_obs, num_states) 
        fill_A(A, prob=prob_A, reward_location=reward_location)
        pA=utils.dirichlet_like(A,scale=1.)
        # pass agent the new A matrix
        my_agent.A=A
        my_agent.pA=pA
        my_agent.reset()
        env.reset(reward_location=reward_location, init_state=init_room, init_tool=init_tool) 
        
        # step agent once with a 'do nothing' action (needed so that qB runs)
        next_action=(0,0,0)
        my_agent.action=np.array(next_action) 
        next_observation = env.step(next_action)
        env.render(title="Run" + str(i+1)+ ". Start", save_in="stickman/"+ str(filecount))
        qs_prev = my_agent.infer_states(next_observation)
        
        filecount += 1
        utility_rank = 0
        infogain_rank = 0
        G_rank = 0
        
        # loop over steps for a single run
        j = 0
        while j < steps_per_run:
            
            qs = my_agent.infer_states(next_observation)
            # this is new method for factorised B and decomposition of G 
            q_pi, G,G1,G2,G3 = my_agent.infer_policies_factorized_expand_G()
            
            # compute lnE (which is the natural log of E vector)
            # E vector in this case is a uniform distribution over policy (so 1/256)
            lnE = np.empty(256)
            lnE.fill(np.log(1/256))
            gamma = 16.0
            
            # plot selected policy with utility vs info gain contributions
            plot_barchart(j+1,i+1,G,G1,G2,G3,filecount)
            
            # get the rank of the chosen policy in G1 and G2+G3
            # we ignore ties in ranking but impact negligible after a few actions
            utility_rank += len(G) - (G1).argsort().argsort()[np.argmax(G)]
            infogain_rank += len(G) - (G2+G3).argsort().argsort()[np.argmax(G)]
            G_rank += len(G) - (G).argsort().argsort()[np.argmax(G)]
            
            # update A and B matrices 
            qA=my_agent.update_A(next_observation)   
            qB=my_agent.update_B(qs_prev)
            qs_prev=qs
            
            # we get policy_idx so can track how Gs develop 
            next_action, policy_idx = my_agent.sample_action() 
            
            assert all(x==next_action[0] for x in next_action), "Something gone wrong. All states should have the same action."

            # get next obs by stepping environment
            next_observation = env.step(next_action)
            
            # print useful information about actions and observations
            print("Actions:", next_action, ". Returned Observations:", next_observation)
            print("Actions:", room_action[int(next_action[2])], x_action[int(next_action[0])],y_action[int(next_action[1])],". Returned Observations:", obs_tool[next_observation[0]],obs_room[next_observation[1]],". Returned reward:", obs_reward[next_observation[2]])
            
            # render gridworld plot
            env.render(title="Run:" + str(i+1)+ ". Step:"+str(j+1) + ". Action:" +str(room_action[int(next_action[0])]), save_in="stickman/"+ str(filecount))
            
            # step loop
            filecount += 1
            j += 1
            
            # keep track of whether reward has been found
            if next_observation[2]==0:
                num_steps_history.append(0)
            else:
                num_steps_history.append(1)
            
            tool_certainty.append([my_agent.B[0][1,0,1,2],my_agent.B[1][1,0,0,2],my_agent.B[0][1,0,1,2]*my_agent.B[1][1,0,0,2]])
        
        # knowledge of tool over a run
        plot_tool_certainty(tool_certainty, reward_location)
        
        # we are just taking the average ranking over a single run        
        G_rank_history.append(G_rank/j)
        utility_rank_history.append(utility_rank/j)
        infogain_rank_history.append(infogain_rank/j)
        
        # plot num_steps to find reward
        plot_num_steps_history(num_steps_history, reward_location)
        
        print(f"finished run {str(i+1)}")
    
    plot_policy_rankings(utility_rank_history, infogain_rank_history, G_rank_history, reward_location)

    print("finished experiment")

def main():
    run_experiment(locations, num_runs, steps_per_run, num_obs, num_states, prob_A, my_agent, env, init_room, init_tool)

if __name__=="__main__":
   main()