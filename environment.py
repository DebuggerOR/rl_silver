import random as rand

HIT = 0
STICK = 1


class Easy21Env:
    def __init__(self, min_card=1, max_card=10, up_bound=21, low_bound=1, dealer_min=17, neg_rate=1/3):
        self.min_card = min_card
        self.max_card = max_card
        self.up_bound = up_bound
        self.low_bound = low_bound
        self.dealer_min = dealer_min
        self.neg_rate = neg_rate

    def init_game(self):
        self.player_sum = self.draw_card(is_sure_black=True)
        self.dealer_sum = self.draw_card(is_sure_black=True)
        print(f'*** init - player: {self.player_sum}, dealer: {self.dealer_sum} ***')

    def draw_card(self, is_sure_black=False):
        red = not is_sure_black and rand.random() < self.neg_rate
        num = rand.randint(self.min_card, self.max_card)
        if red:
            num *= -1
        print(f'{num} card was drawn')
        return num

    def run_dealer(self):
        while self.dealer_sum < self.dealer_min:
            card = self.draw_card()
            self.dealer_sum += card
            print(f'dealer takes {card} with total {self.dealer_sum}')

            if self.is_busted(self.dealer_sum):
                return 1

        if self.dealer_sum > self.player_sum:
            return -1
        elif self.dealer_sum < self.player_sum:
            return 1

        return 0

    def hypo_run_dealer(self):
        dealer_sum = self.dealer_sum
        while dealer_sum < self.dealer_min:
            card = self.draw_card()
            dealer_sum += card
            print(f'if dealer takes {card} with total {dealer_sum}')

            if self.is_busted(dealer_sum):
                return 1

        if dealer_sum > self.player_sum:
            return -1
        elif dealer_sum < self.player_sum:
            return 1
        return 0

    def run_player(self):
        card = self.draw_card()
        self.player_sum += card
        print(f'player takes {card} with total {self.player_sum}')

        if self.is_busted(self.player_sum):
            return -1
        return 0

    def hypo_run_player(self):
        player_sum = self.player_sum
        card = self.draw_card()
        player_sum += card
        print(f'if player takes {card} with total {player_sum}')

        if self.is_busted(player_sum):
            return -1
        return 0

    def is_busted(self, sum):
        return sum < self.low_bound or sum > self.up_bound

    def step(self, action, hypo=False):
        print(f'*** action ({action}) - player: {self.player_sum}, dealer: {self.dealer_sum} ***')

        if action == STICK:
            reward = self.run_dealer() if not hypo else self.hypo_run_dealer()
            terminated = True
        else:
            reward = self.run_player() if not hypo else self.hypo_run_player()
            terminated = (reward != 0)

        print(f'reward: {reward}')
        return reward, terminated

    def get_num_states(self):
        return self.up_bound + 2 - self.low_bound, self.max_card + 2 - self.min_card

    def get_state(self):
        print(f'player sum: {self.player_sum}, dealer init: {self.dealer_sum}')
        return self.player_sum, self.dealer_sum
