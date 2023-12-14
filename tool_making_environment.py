#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Reaching in a grid world environment


__author__: Paul Kinghorn
based on Conor Heins, Alexander Tschantz, Brennan Klein gridworld

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import time



from pymdp.envs import Env


class toolEnv(Env):
    """ 2-dimensional grid-world implementation with 5 actions (the 4 cardinal directions and staying put)."""

    # move actions
    STAY = 0
    MOVE = 1    
    MOVE_ACTIONS = ["STAY", "MOVE"]
    
    # tool actions
    NULL=0
    PICK=1
    DROP=2
    TOOL_ACTIONS= ["Null", "Pick-up", "Drop"]
    
    # tool states
    TOOL_NONE=0
    TOOL_V=1
    TOOL_H=2
    TOOL_HV=3
    TOOL_VH=4
    TOOL_STATES=["Null tool", "V", "H", "HV", "VH"]
    ROOM_0=0
    ROOM_1=1
    ROOM_STATES=["Room0", "Room1"]


    def __init__(self, reduced_obs=False, reward_location=None, init_state=None, init_tool=None, init_reach=None):
        """
        Initialization function for 2-D grid world

        Parameters
        ----------
        shape: ``list`` of ``int``, where ``len(shape) == 2``
            The dimensions of the grid world, stored as a list of integers, storing the discrete dimensions of the Y (vertical) and X (horizontal) spatial dimensions, respectively.
        init_state: ``int`` or ``None``
            Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
            If ``None``, then an initial location will be randomly sampled from the grid.
        """
        
        self.reduced_obs=reduced_obs
        self.n_move_actions = len(self.MOVE_ACTIONS)
        self.n_tool_actions=len(self.TOOL_ACTIONS)
        self._build()
        self.reset(reward_location, init_state, init_tool, init_reach)
        # self.set_init_state(init_state)
        # self.set_init_tool(init_tool)
        # self.set_reward_location(reward_location)
        # self.set_init_reaching_state(init_reach)
        # self.last_move = None
        
        self.stickman_pic=image.imread("img/stickman.png")
        self.reward_pic=image.imread("img/reward.png")
        self.reward_found_pic=image.imread("img/reward_found.png")

    def reset(self, reward_location=None, init_state=None, init_tool=None, init_reach=None):
        """
        Reset the state of the 2-D grid world. In other words, resets the location of the agent, and wipes the current action.

        Parameters
        ----------
        init_state: ``int`` or ``None``
            Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
            If ``None``, then an initial location will be randomly sampled from the grid.

        Returns
        ----------
        self.state: ``int``
            The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
        """
        self.set_init_state(init_state)
        self.set_init_tool(init_tool)
        self.set_reward_location(reward_location)
        self.set_init_reaching_state(init_reach)
        self.last_move = None
        #return self.state

    # def set_state(self, state):
    #     """
    #     Sets the state of the 2-D grid world.

    #     Parameters
    #     ----------
    #     state: ``int`` or ``None``
    #         State of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
    #         If ``None``, then a location will be randomly sampled from the grid.

    #     Returns
    #     ----------
    #     self.state: ``int``
    #         The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
    #     """
    #     self.state = state
    #     return state

    def set_reward_value(self):
        """
        Sets the reward observation of the 2-D grid world to True or False, depening on whether state matches reward location.

        Parameters
        ----------
        state: ``int`` or ``None``
            State of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
            If ``None``, then a location will be randomly sampled from the grid.

        Returns
        ----------
        reward: {0,1}. 
        1 if reward location
        0 if not reward location
        TBD - need to get in a minus reward for creating a new tool
        """
        if self.reach_state==1:
            if self.room_state==self.reward_room and self.tool_state==self.reward_tool:
                reward=1 # reward
            else:
                reward=0 # punish
        else:
            reward=0 # punish
        return reward

    def step(self, action):
        """
        Updates the state of the environment, i.e. the location of the agent, using an action index that corresponds to one of the 5 possible moves.

        Parameters
        ----------
        action: ``int`` 
            Action index that refers to which of the 5 actions the agent will take. Actions are, in order: "UP", "RIGHT", "DOWN", "LEFT", "STAY".

        Returns
        ----------
        state: ``int``
            The new, updated state of the environment, i.e. the location of the agent in grid world after the action has been made. Will be discrete index in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
        """
        

        
        if self.reduced_obs:
            # action is passed in as 0- null, 1- move, 2- pickup 3 - drop.  reach assumed to always be 1
            if action==1:
                move=1
            else:
                move=0
            room_state = self.P[self.room_state][move]
            self.room_state = room_state
            
            if action==2: 
                tool= 1
            elif action==3:
                tool= 2
            else: 
                tool=0
            tool_state = self.R[self.room_state][self.tool_state][tool]
            self.tool_state=tool_state
            
            reach_state=1
            self.reach_state=reach_state
            
            reward=self.set_reward_value()
            self.reward=reward
            
            # dont return reach
            # combine room and tool into a single observation
            combined_state=room_state*5+tool_state
            return [combined_state, reward]
        else:
            
            #######
            ## EITHER coping with a single action and 2 states sending the action (tool and room)
            ### OR 3 states sending that aciton (tool, x_reach, y_reach)
            ## So either: 
            move=action[1]
            tool=action[0]
            
            #### code below introduced to cope with single action- both states send the same aciton
            #### or thinkof as " there is only one action"
            ###check they are the same
            assert move==tool, "This is not a single action setup - receiving different actions for Room and Tool states"
            #now convert the new action into the original action which this environment was written for
            move_convert = {0:0, 1:1, 2:0, 3:0} ##format new action old action so eg new 2pickup = do nothing
            tool_convert = {0:0, 1:0, 2:1, 3:2}
            move=move_convert[move]
            tool=tool_convert[tool]
            ### OR this
            
            ###########################
            
            reach=1 # always reaching
            room_state = self.P[self.room_state][move]
            self.room_state = room_state
            
            tool_state = self.R[self.room_state][self.tool_state][tool]
            self.tool_state=tool_state
    
            reach_state=int(reach)
            self.reach_state=reach_state
    
            reward=self.set_reward_value()
            self.reward=reward
            
            self.last_move = move
            return [tool_state,room_state, reward]



    def render(self, title=None, save_in=None):
        """
        Creates a heatmap showing the current position of the agent and reward in the grid world.
        Also draws a line showing where the agent reaches out to using tool

        all hardcoded for the 6 possible reward rooms and 2 possible agent rooms and 5 possible tool states

        Parameters
        ----------
        title: ``str`` or ``None``
            Optional title for the heatmap.
        """
        coord_lookup = {0:[1,1] , 1: [2,1], 2:[3,0], 3:[3,1], 4:[0,1], 5:[0,0], 6:[1,0], 7:[2,0]}
        H_tool_x1= 1.7 ; H_tool_x2= 2.3; H_tool_y1= 1.4; H_tool_y2= 1.4;
        V_tool_x1= 1.4; V_tool_x2= 1.4; V_tool_y1= 0.7; V_tool_y2= 1.3;
        
        
        cmp = ListedColormap([ 'lavender', 'royalblue'])
        values = np.zeros((2,4))
        for i in [0,1]:
            values[coord_lookup[i][1],coord_lookup[i][0]] = 0
        for i in [2,3,4,5,6,7]:
            values[coord_lookup[i][1],coord_lookup[i][0]] = 1
        fig, ax = plt.subplots()
        # set up grid
        ax.imshow(values,cmap=cmp)
        for i in [0,1]:
            ax.text(coord_lookup[i][0]+.35,coord_lookup[i][1]-.35, i, color="k")
        for i in [2,3,4,5,6,7]:           
            ax.text(coord_lookup[i][0]+.35,coord_lookup[i][1]-.35, i, color="w")
                    
        # show tool locations
        ax.plot([H_tool_x1, H_tool_x2],[H_tool_y1, H_tool_y2], 'k-', lw=3)
        ax.plot([V_tool_x1, V_tool_x2],[V_tool_y1, V_tool_y2], 'k-', lw=3)
        #place reward
        if self.reward==1:
            rewardbox = OffsetImage(self.reward_found_pic, zoom = 0.15)
        else:
            rewardbox = OffsetImage(self.reward_pic, zoom = 0.15)
        rewardab = AnnotationBbox(rewardbox, (coord_lookup[self.reward_location][0],coord_lookup[self.reward_location][1]), frameon = False)
        ax.add_artist(rewardab)
        # place stickman
        stickbox = OffsetImage(self.stickman_pic, zoom = 0.15)
        stickab = AnnotationBbox(stickbox, (coord_lookup[self.room_state][0],coord_lookup[self.room_state][1]), frameon = False)
        ax.add_artist(stickab)
        # add tools to stickman
        
        if self.tool_state==1:
            ax.plot([coord_lookup[self.room_state][0]+.25, coord_lookup[self.room_state][0]+.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]-.95], 'r-', lw=2)
        elif self.tool_state==2:
            #if self.state==0:
            #ax.plot([coord_lookup[self.room_state][0]-.25, coord_lookup[self.room_state][0]-1.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]+.05], 'r-', lw=2)    
            #elif self.state==1:            
            ax.plot([coord_lookup[self.room_state][0]+.25, coord_lookup[self.room_state][0]+1.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]+.05], 'r-', lw=2)        
        elif self.tool_state==3:            
            ax.plot([coord_lookup[self.room_state][0]+.25, coord_lookup[self.room_state][0]+1.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]+.05], 'r-', lw=2)        
            ax.plot([coord_lookup[self.room_state][0]+1.25, coord_lookup[self.room_state][0]+1.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]-.95], 'r-', lw=2)
        elif self.tool_state==4:            
            ax.plot([coord_lookup[self.room_state][0]-.25, coord_lookup[self.room_state][0]-1.25],[coord_lookup[self.room_state][1]-.95, coord_lookup[self.room_state][1]-.95], 'r-', lw=2)        
            ax.plot([coord_lookup[self.room_state][0]-.25, coord_lookup[self.room_state][0]-.25],[coord_lookup[self.room_state][1]+0.05, coord_lookup[self.room_state][1]-.95], 'r-', lw=2)
             
        plt.xticks(np.arange(3,step=1)+.5, color='white')
        plt.yticks(np.arange(1,step=1)+.5, color='w')
        ax.grid(True,color='w', linewidth=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False)

        if title != None:
            plt.title(title,loc="left")
        plt.savefig(save_in)
        plt.show()
        if self.reward==1:
            time.sleep(2)
        

    def set_init_state(self, init_state=None):
        if init_state != None:
            if init_state not in {0,1}:
                raise ValueError("not a valid `init_state`")
            if not isinstance(init_state, (int, float)):
                raise ValueError("`init_state` must be [int/float]")
            self.init_state = int(init_state)
        else:
            self.init_state = np.random.randint(0, 2)
        self.room_state = self.init_state
        
    def set_init_tool(self, init_tool=None):
        if init_tool != None:
            if init_tool > 4 or init_tool < 0:
                raise ValueError("`init_tool` is greater than number of tools")
            if not isinstance(init_tool, (int, float)):
                raise ValueError("`init_tool` must be [int/float]")
            self.init_tool = int(init_tool)
        else:
            self.init_tool = 0 ################## forced to be no tool
        self.tool_state = self.init_tool
        
    def set_reward_location(self, reward_location=None):
        if reward_location != None:
            if reward_location not in {2,3,4,5,6,7}:
                raise ValueError("`reward location` is greater than number of states")
            if not isinstance(reward_location, (int, float)):
                raise ValueError("`reward_location` must be [int/float]")
            self.reward_location = int(reward_location)
            # reward_location is the room the reward is in. from this we simply lookup what room and tool of the agent needs to be
            lookup = {2:[1,3], 3:[1,2], 4:[0,2], 5:[0,3], 6:[0,1], 7:[1,1]} # so eg if reward_location=2, then agent must be in room 1 with tool 3
            
            self.reward_room=lookup[reward_location][0]
            self.reward_tool=lookup[reward_location][1]
            # these are then used in set_reward_value() 
        else:
            self.reward_location = 2 ########### forced to be 2
        
    def set_init_reaching_state(self, init_reach=None):
        if init_reach != None:
            if init_reach not in {0,1}:
                raise ValueError("`init_reach` is greater than number of reaching states")
            if not isinstance(init_reach, (int, float)):
                raise ValueError("`init_reach` must be [int/float]")
            self.init_reach = int(init_reach)
        else:
            self.init_reach = 0 ################## forced to be no tool
        self.reach_state = self.init_reach


    def _build(self):
        P = {}
        # only two room states - move action changes which one we move to 
        P[0] = {a: [] for a in range(self.n_move_actions)}
        P[1] = {a: [] for a in range(self.n_move_actions)}
        P[0][self.STAY]=0
        P[0][self.MOVE]=1
        P[1][self.STAY]=1
        P[1][self.MOVE]=0
        self.P = P # rules for movement. Lists, for each state, the next state according to action
        
        # 5 tool states - pickup action changes it depending on what current room is and current tool 
        R = [{},{}]
        for i in range(len(R)):
            R[i][0] = {a: [] for a in range(self.n_tool_actions)}
            R[i][1] = {a: [] for a in range(self.n_tool_actions)}
            R[i][2] = {a: [] for a in range(self.n_tool_actions)}
            R[i][3] = {a: [] for a in range(self.n_tool_actions)}
            R[i][4] = {a: [] for a in range(self.n_tool_actions)}
        # so eg R[0][1][4] descriebs how tool states changes if in room 0, with tool 1 and tool action 4
    
        for i in range(5):
            R[0][i][self.NULL]=i
            R[0][i][self.DROP]=0
            R[1][i][self.NULL]=i
            R[1][i][self.DROP]=0
        R[0][self.TOOL_NONE][self.PICK]=self.TOOL_V
        R[0][self.TOOL_V][self.PICK]=self.TOOL_V
        R[0][self.TOOL_H][self.PICK]=self.TOOL_HV
        R[0][self.TOOL_HV][self.PICK]=self.TOOL_HV
        R[0][self.TOOL_VH][self.PICK]=self.TOOL_VH
        
        R[1][self.TOOL_NONE][self.PICK]=self.TOOL_H
        #R[1][self.TOOL_V][self.PICK]=self.TOOL_VH
        R[1][self.TOOL_V][self.PICK]=self.TOOL_HV
        R[1][self.TOOL_H][self.PICK]=self.TOOL_H
        R[1][self.TOOL_HV][self.PICK]=self.TOOL_HV
        R[1][self.TOOL_VH][self.PICK]=self.TOOL_VH
               
        self.R = R # rules for reaching. Lists, for each state, the reached state according to the current tool
        

    def get_init_state_dist(self, init_state=None):
        init_state_dist = np.zeros(self.n_states)
        if init_state == None:
            init_state_dist[self.init_state] = 1.0
        else:
            init_state_dist[init_state] = 1.0

    def get_transition_dist(self):
        B = np.zeros([self.n_states, self.n_states, self.n_control])
        for s in range(self.n_states):
            for a in range(self.n_control):
                ns = int(self.P[s][a])
                B[ns, s, a] = 1
        return B

    def get_likelihood_dist(self):
        A = np.eye(self.n_observations, self.n_states)
        return A

    def sample_action(self):
        return np.random.randint(self.n_control)

    @property
    def position(self):
        """ @TODO might be wrong w.r.t (x & y) """
        return np.unravel_index(np.array(self.state), self.shape)
    
    @property
    def reward_position(self):
        """ @TODO might be wrong w.r.t (x & y) """
        return np.unravel_index(np.array(self.reward_location), self.shape)

    @property
    def reached_position(self):
        """ @TODO might be wrong w.r.t (x & y) """
        return np.unravel_index(np.array(self.reached_state), self.shape)
