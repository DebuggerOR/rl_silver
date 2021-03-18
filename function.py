
import numpy as np
import random as rand
import matplotlib.pyplot as plt


def feature_vec(s0,s1,a):
    v = np.zeros((3,6,2))

    dealer_ranges = [range(1,4), range(4,7), range(7,10)]
    player_ranges = [range(1,6), range(4,9), range(7,12), range(10,15), range(13,18), range(16,21)]

    for i in range(len(dealer_ranges)):
        if s1 in dealer_ranges[i]:
            for j in range(len(player_ranges)):
                if s0 in player_ranges[j]:
                    v[i,j,a] = 1

    return v


class FunctionControl:
    def __init__(self, env, actions, num_episodes=100000, decay=0.5, l=0.5, epsilon=0.05, alpha=0.01):
        self.env = env

        self.num_episodes = num_episodes
        self.actions = actions

        self.w = np.zeros((3, 6, 2))

        self.decay = decay
        self.l = l
        self.epsilon = epsilon
        self.alpha = alpha

    def greedy(self):
        state = self.env.get_state()
        action = self.actions[np.argmax([np.dot(self.w,feature_vec(state[0],state[1],a)) for a in self.actions])]
        return action

    def epsilon_greedy(self):
        state = self.env.get_state()

        # explore
        if rand.random() < self.epsilon:
            action = rand.choice(self.actions)
        # exploit
        else:
            action = self.actions[np.argmax([np.vdot(self.w,feature_vec(state[0],state[1],a)) for a in self.actions])]

        return action

    def control(self):
        for i in range(self.num_episodes):
            print(f'*** run episode {i} ***')
            self.env.init_game()

            # run episode
            E = np.zeros(self.w.shape)
            next_state = self.env.get_state()
            next_action = self.epsilon_greedy()
            while True:
                state = next_state
                action = next_action

                fv = feature_vec(state[0], state[1], action)
                E += fv

                reward, terminated = self.env.step(action)

                delta = reward - np.vdot(self.w,fv)

                if terminated:
                    self.w += self.alpha * delta * E
                    break

                next_state = self.env.get_state()
                next_action = self.epsilon_greedy()

                nfv = feature_vec(next_state[0],next_state[1],next_action)
                Q_a = np.vdot(self.w,nfv)

                delta += self.decay * Q_a
                self.w += self.alpha * delta * E
                E *= self.decay * self.l

        s0, s1 = 21,11
        V1 = np.array([[np.max([np.vdot(self.w,feature_vec(x,y,a)) for a in self.actions])
                        for x in range(s0)]
                       for y in range(s1)])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(range(s0), range(s1))
        surf = ax.plot_surface(X, Y, V1, cmap='OrRd', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(f'Function {self.num_episodes} episodes')
        plt.show()
