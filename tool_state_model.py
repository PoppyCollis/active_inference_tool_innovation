# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:24:18 2023

UNFACTORISED TOOL STATE MODEL
EXPERIMENTS: TOOL USE and TOOL DISCOVERY 

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


def build_A_tool(A, prob):
    """
    Returns
    -------
    A : A[0] filled of A matrix

    """
    # A[0].shape = 2 * 4 * 2
    for i in range(A[0].shape[2]): # 2
        non_prob=(1-prob)/(A[0].shape[0]-1)
        A[0][:,:,i] = np.ones((A[0].shape[0], A[0].shape[1]))*non_prob
        np.fill_diagonal(A[0][:,:,i], prob)
                 
    return A 

def build_A_room(A, prob):
    """
    Returns
    -------
    A : A[1] filled of A matrix

    """
    for i in range(A[1].shape[1]):
        non_prob=(1-prob)/(A[1].shape[0]-1)
        A[1][:,i,:] = np.ones((A[1].shape[0], A[1].shape[2]))*non_prob
        np.fill_diagonal(A[1][:,i,:], prob)
    return A

def build_A_reward(A, reward_location, prob):
    """
    Returns
    -------
    A : A[2] filled of A matrix

    """
    # if we want VH and HV to differ then 5:[0,4], else: 5:[0,3]
    lookup = {2:[1,3], 3:[1,2], 4:[0,2], 5:[0,3], 6:[0,1], 7:[1,1]} # [room, tool]
    reward_room = lookup[reward_location][0]
    reward_tool = lookup[reward_location][1]
    # if reward location fixed, tell agent what room and tool states combinations will get reward
    non_prob=(1-prob)/(A[2].shape[0]-1)
    A[2][0,:,:] = prob; 
    A[2][1,:,:] = non_prob;
    A[2][0,reward_tool,reward_room] = non_prob; 
    A[2][1,reward_tool,reward_room] = prob
    
    return A
     
def fill_A(A, reward_location, prob=1):
    build_A_room(A, prob)
    build_A_tool(A, prob)
    build_A_reward(A, reward_location, prob)
    return A


def build_B_tool(B, fully_known, prob):
    """
    Note
    -------
    Tool state changes from 2nd dmiension to first
    e.g. B[0][3,2,0,1] is: action 1 (pickup) in room 0 changes tool state 2 to tool state 3
    
    Returns
    -------
    B : fills in B[0]
    """
    if fully_known:
        # tool state null action:
        # in room 0
        non_prob = 1-prob
        B[0][:,:,0,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,0], prob)
        # in room 1
        B[0][:,:,1,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,0], prob)
    
        # tool state move action:
        # in room 0
        B[0][:,:,0,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,1], prob)
        # in room 1
        B[0][:,:,1,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,1], prob)    
            
        # tool state pick-up action:
        # in room 0
        B[0][:,:,0,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][1,0,0,2] = prob
        B[0][1,1,0,2] = prob
        B[0][3,2,0,2] = prob
        B[0][3,3,0,2] = prob
        #B[0][4,4,0,2] = prob
        # in room 1
        B[0][:,:,1,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][2,0,1,2] = prob
        #B[0][4,1,1,2] = prob
        B[0][2,2,1,2] = prob
        B[0][3,3,1,2] = prob
        B[0][3,1,1,2] = prob
       # B[0][4,4,1,2] = prob 
                
        # tool state drop action:
        # drops everything and they go back to original starting place
        # in room 0
        B[0][:,:,0,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,0,3] = prob
        # in room 1
        B[0][:,:,1,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,1,3] = prob
        
    else:
        # all actions are totally unknown 
        # i.e. B[:,:,: for B[0] - each factor list factor is the same
        for i in range(B[0].shape[2]):
            for j in range(B[0].shape[3]):
                B[0][:,:,i,j] = np.ones((B[0].shape[0],B[0].shape[1])) / B[0][:,:,i,j].shape[0]   
                
    return B

def build_B_room(B, fully_known, prob):
    """
    Returns
    -------
    B : fills in B[1]
    """
    if fully_known:
        non_prob = (1-prob)/(B[0].shape[0]-1)
        # room state, null action
        B[1][:,:,0] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,0], prob)
        # room state, move action
        B[1][:,:,1] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][0,1,1] = prob
        B[1][1,0,1] = prob
        # room state, pick-up action
        B[1][:,:,2] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,2], prob)
        # room state, drop-off action
        B[1][:,:,3] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,3], prob)
        
    else:
        # all actions are totally unknown 
        # i.e. each B[:,:,i] is the same everywhere adding up to 1.0 for B[1]
        for i in range(B[1].shape[2]):
            B[1][:,:,i] = np.ones((B[1].shape[0],B[1].shape[1])) / B[1][:,:,i].shape[0]   
        
        
    return B
        
def fill_B(B, fully_known=False, prob=1):
    B = build_B_tool(B, fully_known, prob)
    B = build_B_room(B, fully_known, prob)
    return B

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
state_tool = ["None", "V", "H", "HV"]

# controls
room_action = ["Null", "Move", "Pick-up", "Drop"]
tool_action = ["Null", "Move", "Pick-up", "Drop"]
combined_action_list=[room_action, tool_action]

# observation modality dimensions
num_obs = [len(obs_tool), len(obs_room),len(obs_reward)] 
# hidden state factor dimensions
num_states = [len(state_tool),len(state_room)] 
# control state factor dimensions
num_controls = [len(tool_action),len(room_action)] 


# B_factor_list says the states each state depends on
# 0 (tool state), 1 (room state)
B_factor_list=[[0,1],[1]]

# set dummy reward location to allow us to initialise the agent  
reward_location = 3

# create observation likelihood (A matrix)
A = utils.initialize_empty_A(num_obs, num_states) 
prob_A = 1. # confidence in A matrix
fill_A(A, reward_location=reward_location, prob=prob_A)
dir_scale = 1.
pA=utils.dirichlet_like(A,scale=dir_scale)

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
locations = [6,6,6,6,3,3,3,3,4,4,4,4,7,7,7,7,2,2,2,2,5,5,5,5] # changing reward locations
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
        next_action=(0,0)
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
            print("Actions:", room_action[int(next_action[1])], tool_action[int(next_action[0])],". Returned Observations:", obs_tool[next_observation[0]],obs_room[next_observation[1]],". Returned reward:", obs_reward[next_observation[2]])
            
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
            
            tool_certainty.append([my_agent.B[0][2,0,1,2],my_agent.B[0][1,0,0,2],my_agent.B[0][3,1,1,2]])
        
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