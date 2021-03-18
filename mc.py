
import numpy as np
import random as rand
import matplotlib.pyplot as plt


class MCControl:
    def __init__(self, env, num_states, actions, N_0=100, num_episodes=1_000_000):
        self.env = env

        self.num_episodes = num_episodes
        self.actions = actions
        self.N_0 = N_0

        self.Q = np.zeros((num_states[0], num_states[1], len(actions)))
        self.Ns = np.zeros((num_states[0], num_states[1]))
        self.Nsa = np.zeros((num_states[0], num_states[1], len(actions)))

    def greedy(self):
        state = self.env.get_state()
        action = self.actions[np.argmax([self.Q[state[0], state[1], a] for a in self.actions])]
        return action

    def epsilon_greedy(self):
        state = self.env.get_state()
        epsilon = self.N_0 / (self.N_0 + self.Ns[state[0], state[1]])

        print(f'epsilon is {epsilon}')

        # explore
        if rand.random() < epsilon:
            action = rand.choice(self.actions)
        # exploit
        else:
            action = self.actions[np.argmax([self.Q[state[0], state[1], a] for a in self.actions])]

        print(f'Q[{state[0]},{state[1]}] is {self.Q[state[0], state[1], :]}, chosen action is {action}')
        return action

    def control(self):
        for i in range(self.num_episodes):
            print(f'*** run episode {i} ***')
            self.env.init_game()
            reward = None
            sa = []    # state, action

            # run episode
            terminated = False
            while not terminated:
                state = self.env.get_state()
                action = self.epsilon_greedy()

                self.Ns[state[0], state[1]] += 1
                self.Nsa[state[0], state[1], action] += 1

                sa.append([state, action])

                reward, terminated = self.env.step(action)

            # update Q
            mean = 0
            for (s, a) in sa:
                alpha = 1 / self.Nsa[s[0], s[1], a]
                gap = (reward - self.Q[s[0], s[1], a])
                print(f'alpha is {alpha}, gap is {gap}')

                self.Q[s[0], s[1], a] += alpha * gap
                mean += abs(reward - self.Q[s[0], s[1], a])

            print(f'--- episode {i}, reward {reward}, mean error {mean / len(sa)} ---')

        s0, s1, _ = self.Q.shape
        V1 = np.array([[np.max(self.Q[x, y]) for x in range(s0)] for y in range(s1)])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(range(s0), range(s1))
        surf = ax.plot_surface(X, Y, V1, cmap='OrRd', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(f'MC {self.num_episodes} episodes')
        plt.show()
