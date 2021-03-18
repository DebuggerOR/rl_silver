
import numpy as np
import random as rand
import matplotlib.pyplot as plt


class TDControl:
    def __init__(self, env, num_states, actions, N_0=10, num_episodes=100000, decay=0.5):
        self.env = env

        self.num_episodes = num_episodes
        self.actions = actions
        self.N_0 = N_0

        self.Q = np.zeros((num_states[0], num_states[1], len(actions)))
        self.Ns = np.zeros((num_states[0], num_states[1]))
        self.Nsa = np.zeros((num_states[0], num_states[1], len(actions)))

        self.decay = decay

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
                self.Q[state[0], state[1], action] += alpha * gap

            print(f'--- episode {i} ---')

        s0, s1, _ = self.Q.shape
        V1 = np.array([[np.max(self.Q[x, y]) for x in range(s0)] for y in range(s1)])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(range(s0), range(s1))
        surf = ax.plot_surface(X, Y, V1, cmap='OrRd', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(f'TD-{0} {self.num_episodes} episodes')
        plt.show()
