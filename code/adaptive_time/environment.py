import numpy as np


class MountainCar(object):
    def __init__(self, horizon_sec=200.0, dt_sec=1.0, es: int=50):
        self.horizon_sec = horizon_sec
        self.dt_sec = dt_sec
        self.es = es
        self.horizon = int(horizon_sec / dt_sec)
        self.means = np.array([0, 1.0])
        self.reset()

    def reset(self, episode_i: int=50):
        """Reset the environment to the initial state, return that state."""
        # self.pos = np.random.uniform(low=-1.2,high=0.6)
        # self.pos = np.random.uniform(low=-1.2,high=0.3)
        # self.vel = np.random.uniform(low=-0.07,high=0.07)
        if episode_i < self.es:
            print("ES")
            self.pos = np.random.uniform(low=-1.2,high=0.6)
            self.vel = np.random.uniform(low=-0.07,high=0.07)
        else:
            self.pos = -0.5
            self.vel = 0.0
        self.done = False
        self.h = 0
        self.h_disc = -1
        return [self.pos, self.vel]

    def step(self, action):
        action = [-1, 0, 1][action]
        self.h += self.dt_sec
        self.h_disc += 1
        self.vel = max(
            min(
                (
                    self.vel
                    + self.dt_sec * (0.001 * action + -0.0025 * np.cos(3 * self.pos))
                ),
                0.07,
            ),
            -0.07,
        )

        if self.pos >= 0.6:
            # At goal; stay there.
            self.pos = 0.6
            reward = 0.0
        else:
            # Not at goal; move.
            self.pos = max(self.pos + self.dt_sec * self.vel, -1.2)
            reward = -1.0

        if self.h + self.dt_sec > self.horizon_sec:
            self.done = True

        return reward, np.array([self.pos, self.vel]), (self.h, self.h_disc), self.done

    # Code for broadcasting Mountain Car, might be useful later.

    """

    def step_broadcast(self, s, action, n, var):
        self.h += 1
        
        pos = s[0,:]
        vel = s[1,:]

        noise = np.random.normal(size=len(pos)) * var
        
        vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
        vel = np.where(vel <= -0.07, -0.07, vel)
        vel = np.where(vel >= 0.07, 0.07, vel)
        
        #vel_top_idx = np.where(vel >= 0.07)
        #vel[vel_bottom_idx] = -0.07  
        #vel[vel_top_idx] = 0.07
        
        cost = np.zeros(n)
        
        pos = pos + vel + noise

        pos = np.where(pos <= -1.2, -1.2, pos)
        pos = np.where(pos >= 0.6, 0.6, pos)
        
        if self.h != self.horizon_sec - 1:
            s_ = np.array([pos,vel])
            return cost, s_
        else:
            cost = np.where(pos >= 0.6, np.random.binomial(n=1,p=self.means[0]), np.random.binomial(n=1,p=self.means[1])) #first arguement is for the successful trajectory 
            s_ = np.array([None] * n)
            return cost, s_
        

    def step_broadcast_eval(self, s, action, n, var):
        self.h += 1
        
        pos = s[0,:]
        vel = s[1,:]

        noise = np.random.normal(size=len(pos)) * var
        
        vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
        vel = np.where(vel <= -0.07, -0.07, vel)
        vel = np.where(vel >= 0.07, 0.07, vel)
        
        #vel_top_idx = np.where(vel >= 0.07)
        #vel[vel_bottom_idx] = -0.07  
        #vel[vel_top_idx] = 0.07
        
        cost = np.zeros(n)
        
        pos = pos + vel + noise

        pos = np.where(pos <= -1.2, -1.2, pos)
        pos = np.where(pos >= 0.6, 0.6, pos)
        
        if self.h != self.horizon_sec - 1:
            s_ = np.array([pos,vel])
            return cost, s_
        else:
            cost = np.where(pos >= 0.6, self.means[0], self.means[1]) #first argument is for the successful trajectory 
            s_ = np.array([None] * n)
            return cost, s_
            
"""


def create_trajectories(num_trajectories, policy, horizon_sec, dt_sec):
    """Returns a list of trajectories generated by policy.

    Each trajectory is a tuple of (states, rewards), where

    * states is the list of states visited by the policy before the set
        timeout (horizon_sec). The list is roughly length `horizon_sec / dt_sec`.
    * rewards is the list of rewards received by the policy at each state.
        This list is one shorter than states.
    """
    trajectories = []
    env = MountainCar(horizon_sec, dt_sec)
    for _ in range(num_trajectories):
        rewards = []
        states = []

        state = env.reset()
        states.append(state)

        is_done = False
        while not is_done:
            reward, state, _, is_done = env.step(policy(state))
            rewards.append(reward)
            states.append(state)
        trajectories.append((states, rewards))

    return trajectories
