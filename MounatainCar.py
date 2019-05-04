import sys
import gym 
import time
import numpy as np
import csv
from gym import wrappers

class Q_learnning(object):
    def __init__(self,env_str="MountainCar-v0",f_probability=0.5, max_epi=10000, step_by_epi=200, goal_ave=-150, digitized_num=60):
        self.env = gym.make(env_str)
        self.first_probability = f_probability
        self.max_episodes = max_epi
        self.step_by_episode = step_by_epi
        self.goal_ave = goal_ave
        self.digitized_num = digitized_num
        self.q_table = np.random.uniform(low=-1,high=1,size=(digitized_num**2,self.env.action_space.n))
        self.review_num = 10
        self.reward_of_episode = 0
        self.reward_ave = np.full(self.review_num,0)
        self.ndim_obs = self.env.observation_space.shape[0] 
        self.gamma = 0.99
        self.alhpa = 0.6
        self.learining_is_done = False
        self.do_render = False
        self.bin_parm = []
        for i in range(self.ndim_obs):
            self.bin_parm.append(np.linspace(self.env.observation_space.low[i],self.env.observation_space.high[i],self.digitized_num+1)[1:-1])
        

    def digitized(self,observation,digitized_num):
        state = 0
        for i in range(self.ndim_obs):
            state += np.digitize(observation[i],self.bin_parm[i]) * (self.digitized_num) ** i
        return state


    def dicide_action(self,next_state,episode):
        epsilon = self.first_probability * (1/(episode+1))
        if epsilon <= np.random.uniform(0,1):
            next_action = np.argmax(self.q_table[next_state])
        else:
            next_action = np.random.choice(range(self.ndim_obs))
        return next_action


    def update_Qtable(self,q_table,state,action,reward,next_state):
        next_max_q = max(q_table[next_state])
        q_table[state,action] = (1-self.alhpa) * q_table[state,action] + self.alhpa * (reward + self.gamma * next_max_q)
        return q_table


    def run(self):
        for episode in range(self.max_episodes):
            self.obs = self.env.reset() 
            self.state = self.digitized(self.obs,self.digitized_num)  
            action = np.argmax(self.q_table[self.state])
            self.reward_of_episode = 0 

            for _ in range(self.step_by_episode):
                if self.learining_is_done:
                    self.env.render()

                self.obs,reward,done,info = self.env.step(action)
                self.reward_of_episode += reward

                next_state = self.digitized(self.obs,self.digitized_num)
                self.q_table = self.update_Qtable(self.q_table,self.state,action,reward,next_state)
                action = self.dicide_action(next_state,episode) 
                self.state = next_state
                
                if done:
                    self.reward_ave = np.hstack((self.reward_ave[1:],self.reward_of_episode))
                    print("episode {}, reward {}".format(episode+1,self.reward_of_episode))

                    if self.learining_is_done == 1:
                        pass
                    break

            if (self.reward_ave.mean() >= self.goal_ave) and episode > self.review_num:
                print("Episode %d train agent successfuly!" % episode)
                self.learining_is_done = 1
                if self.do_render == 0:
                    print(self.q_table)
                    with open('./csv/file.csv', 'wt') as f:
                        writer = csv.writer(f)
                        writer.writerows(self.q_table)
                    self.do_render = 1
        self.env.close()

if __name__ == "__main__":
    sim = Q_learnning() 
    sim.run()
    sim.env.close()
