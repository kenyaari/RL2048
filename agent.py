import logging
from collections import defaultdict
import numpy as np
from scipy.special import softmax
from environment import Environment
from game import TwntyFrtyEight

# logging config
logging.basicConfig(level=logging.INFO, 
                    format="%(message)s",
                    handlers=[
                        logging.FileHandler("agent.log"),
                        logging.StreamHandler()],
                    )

class ActionValue:
    def __init__(self, environ: Environment):
        self.environ = environ
        self.q_table = [[]]
            
    def __call__(self, state: int, action: int, weight:np.ndarray=None):
        if self.environ.is_terminal_state(state): return 0
        if weight is None:
            return self.q_table[state][action]
        else:
            X = self.environ.get_feature_vector(state, action)
            return weight.dot(X)

class Agent:
    def __init__(self, environ: Environment):
        self.environ = environ
        self.q = ActionValue(environ)
        # Initialize weight to 0 vector
        self.w = np.zeros(len(self.environ.get_feature_vector(0, 0)))
    
    def find_optimal_weight(self, alpha, discount_factor=1, tolerance=1e-3, alpha_decay=True):
        '''
        Episodic Semi-gradient Sarsa for Estimating q_hat = q_star
        algorithm from page 244 of Sutton Barto 2nd edition
        '''
        def show_progress():
            logging.info(TwntyFrtyEight.state_to_board(S_prime))
            logging.info(f'{episode_count=} {update_count=} \n{self.w}')
            sorted_stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}
            logging.info(f'{sorted_stats} win_rate={stats[2048]/episode_count:.2%} average_steps={update_count/episode_count:.2f}')
        stats = defaultdict(int)
        
        # initialize
        policy = self.softmax_policy
        
        update_count = 0
        episode_count = 0
        while True:
            episode_count += 1
            last_w = np.copy(self.w)
            learning_rate = alpha/(update_count+1) if alpha_decay else alpha
            S = self.environ.get_initial_state()
            A = policy(S)
            while True:
                S_prime = self.environ.transition(S, A)
                R = self.environ.reward(S, A, S_prime)
                grad = self.environ.get_feature_vector(S, A)
                if self.environ.is_terminal_state(S_prime):
                    self.w = self.w + learning_rate*(R - self.q(S, A, self.w))*grad
                    update_count += 1
                    break
                A_prime = policy(S_prime)
                self.w = self.w + learning_rate*(R + discount_factor*self.q(S_prime, A_prime, self.w) - self.q(S, A, self.w))*grad
                update_count += 1
                S = S_prime
                A = A_prime
            
            # Record episode stats
            stats[self.environ.get_state_status(S_prime)] += 1
            
            # Print progress every 100 episodes
            if episode_count%10 == 0: show_progress()
            if np.linalg.norm(last_w-self.w) < tolerance: break
            
        logging.info(f'------------------------Final convergence------------------------')
        show_progress()
        logging.info(f'-----------------------------------------------------------------')
        
    def random_policy(self, s: int) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        return valid_actions[self.environ.rng.integers(len(valid_actions))]
    
    def greedy_policy(self, s: int) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        values = np.array([self.q(state=s, action=a, weight=self.w) for a in valid_actions])
        best_action = valid_actions[self.environ.rng.choice(np.flatnonzero(values == values.max()))]
        return best_action
    
    def epsilon_greedy_policy(self, s: int, epsilon=0.1) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        if self.environ.rng.random(1) < epsilon:
            return valid_actions[self.environ.rng.integers(len(valid_actions))]
        else:
            values = np.array([self.q(state=s, action=a, weight=self.w) for a in valid_actions])
            best_action = valid_actions[self.environ.rng.choice(np.flatnonzero(values == values.max()))]
            return best_action
    
    def softmax_policy(self, s: int) -> int:
        valid_actions = self.environ.get_valid_actions(s)
        action_values = [self.q(state=s, action=a, weight=self.w) for a in valid_actions]
        return valid_actions[self.environ.rng.choice(len(action_values), p=softmax(action_values))]