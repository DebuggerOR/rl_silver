
import numpy as np
import random as rand
import matplotlib.pyplot as plt


class TDLControl:
    def __init__(self, env, num_states, actions, N_0=10, num_episodes=10000, decay=0.5, l=0.5):
        self.env = env

        self.num_episodes = num_episodes
        self.actions = actions
        self.N_0 = N_0

        self.Q = np.zeros((num_states[0], num_states[1], len(actions)))
        self.Ns = np.zeros((num_states[0], num_states[1]))
        self.Nsa = np.zeros((num_states[0], num_states[1], len(actions)))

        self.decay = decay
        self.l = l

    def greedy(self):
        state = self.env.get_state()
        action = self.actions[np.argmax([self.Q[state[0], state[1], a] for a in self.actions])]
        return action

    def epsilon_greedy(self):
        state = self.env.get_state()
        epsilon = self.N_0 / (self.N_0 + self.Ns[state[0], state[1]])

        # explore
        if rand.random() < epsilon:
            action = rand.choice(self.actions)
        # exploit
        else:
            action = self.actions[np.argmax([self.Q[state[0], state[1], a] for a in self.actions])]

        return action

    def control(self):
        for i in range(self.num_episodes):
            print(f'*** run episode {i} ***')
            self.env.init_game()

            # run episode
            E = np.zeros(shape=self.Q.shape)
            next_state = self.env.get_state()
            next_action = self.epsilon_greedy()
            while True:
                state = next_state
                action = next_action

                self.Ns[state[0], state[1]] += 1
                self.Nsa[state[0], state[1], action] += 1

                reward, terminated = self.env.step(action)

                alpha = 1 / self.Nsa[state[0], state[1], action]

                if terminated:
                    self.Q[state[0], state[1], action] += alpha * (reward - self.Q[state[0], state[1], action])
                    break

                next_state = self.env.get_state()
                next_action = self.epsilon_greedy()

                gap = (reward + self.decay * self.Q[next_state[0], next_state[1], next_action]
                       - self.Q[state[0], state[1], action])

                E[state[0], state[1], action] += 1

                for i in range(self.Ns.shape[0]):
                    for j in range(self.Ns.shape[1]):
                        for a in self.actions:
                            self.Q[state[0], state[1], a] += alpha * E[i,j,a] * gap
                            E[i,j,a] *= self.decay * self.l

            print(f'--- episode {i} ---')

        s0, s1, _ = self.Q.shape
        V1 = np.array([[np.max(self.Q[x, y]) for x in range(s0)] for y in range(s1)])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(range(s0), range(s1))
        surf = ax.plot_surface(X, Y, V1, cmap='OrRd', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(f'TD-{self.l} {self.num_episodes} episodes')
        plt.show()
