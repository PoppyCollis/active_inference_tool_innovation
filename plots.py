#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:31:35 2023

Useful plotting functions for metatool experiments

@author: pfkin + pzc
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})


def plot_barchart(i,ii,G,G1,G2,G3,filecount):
        ## plots the EFE components and total EFE for each policy, at a particlar timestep
        # and save to a file so can later construct a video of evolution
        #plt.rcParams.update({'font.size': 10})
        plt.figure()
        bottom=np.zeros(len(G))
        colors=["lightgreen", "green", "tomato"]
        for count,G_factor in enumerate([G2,G3]):
            #plt.bar(np.arange(len(G)),G_factor, width= 1)
            plt.bar(np.arange(len(G)),G_factor, width= 1, bottom=bottom, color=colors[count])
            bottom+=G_factor
            
        plt.bar(np.arange(len(G)),G1, width= 1,color=colors[2])
        
        plt.plot(np.arange(len(G)),G, linewidth= 1,color="k")
        plt.scatter(np.argmax(G),G[np.argmax(G)],  marker="o", facecolors='none',s=500, color="k",zorder=10)
        #plt.title("Run"+str(ii)+". Action" + str(i))
        plt.ylim([-110,60])
        plt.xlabel("Policy number")
        plt.ylabel("$\mathcal{G}$ Components")
        #plt.savefig("G_evol_6_new/"+ str(filecount))
        plt.show()
        
def entropy_of_distribution(probabilities, base=2):
    """Arguments:
            probabilities: a probability distribution
       returns:
            HX:  float, the entropy of the passed distribution"""
    probabilities = np.array(probabilities)
    # work in units of bits by default as base = 2
    H = -np.sum(probabilities*np.log(probabilities,where=probabilities>0))/(np.log(base))
    
    return H

def plot_entropy(q_entropy_full_hist):
    for i in range(len(q_entropy_full_hist)):
        plt.plot(np.linspace(0, len(q_entropy_full_hist[i]), len(q_entropy_full_hist[i])), q_entropy_full_hist[i], label=f"Run {i}" )
    plt.title("entropy of q_pi distribution over runs")
    plt.legend()
    plt.show()
    
    
def plot_tool_certainty(tool_certainty, reward_location):
    tool_certainty_np=np.asarray(tool_certainty).T
    colors=[[],
            [],
            ["blue", "blue", "red", "black"], 
            ["red", "silver", "grey", "black"],
            ["red", "silver", "grey", "black"],
            ["grey", "blue", "black", "red"],
            ["silver","red", "grey", "black"],
            ["silver","red", "grey", "black"]]
    labels=["H", "V", "HV", "VH"]
    plt.figure()
    for tool in range(len(tool_certainty_np)):
        plt.plot(tool_certainty_np[tool,:], color=colors[reward_location][tool],label=labels[tool])    
    plt.title("Knowledge of how to create tools. \nReward in room" +str (int(reward_location)))
    plt.xlabel("Step")
    plt.ylabel("Knowledge of how to create tool")
    plt.legend()
    plt.ylim([0,1])
    plt.show()
    
def plot_policy_rankings(utility_rank_history, infogain_rank_history, G_rank_history, reward_location):
    fig,ax=plt.subplots()
    plt.plot(G_rank_history[:100], color="green", marker='x', markersize=10, label="G rank")
    plt.plot(utility_rank_history[:100],"bo-", label="Utility rank")
    plt.plot(infogain_rank_history[:100],"ro-", label="Info gain rank")
    plt.title("Reward in room:"+ str(reward_location) + "\nRanks of each chosen policy (average per run)." )
    #plt.title("Reward moves from room 2 to 3 and back to 2\nRanks of each chosen policy (average per run)." )
    ### shows when we move room
    # if ii==0:
    #     plt.plot([33,33],[0,1],"k--")
    #     plt.plot([66,66],[0,1],"k--")
    plt.legend()
    plt.xlabel("Run")
    plt.ylabel("Rank")
    ax.invert_yaxis()
    plt.show()
    
def plot_num_steps_history(num_steps_history, reward_location):
    plt.figure()
    plt.plot(num_steps_history,"bo-")
    plt.title("History of num_steps on each run. Reward in room:"+ str(reward_location))
    plt.xlabel("Step (10 per run)")
    plt.ylabel("Was reward found")
    plt.show()