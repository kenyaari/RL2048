from environment import Environment

class Episode:
    def __init__(self, environment: Environment, policy, max_length=10000):
        self.state_history = []
        self.action_history = []
        self.reward_history = [None]
        
        s = environment.get_initial_state()
        self.state_history.append(s)
        
        t = 0
        while not environment.is_terminal_state(s) and t < max_length:
            a = policy(s)
            self.action_history.append(a)
            s_prime = environment.transition(s, a)
            r = environment.reward(s, a, s_prime)
            self.reward_history.append(r)
            t += 1
            s = s_prime
            self.state_history.append(s)
        
        self.length = len(self.action_history)
            
    def state_at(self, t):
        if 0 <= t < len(self.state_history):
            return self.state_history[t]
        else:
            return None
    
    def action_at(self, t):
        if 0 <= t < len(self.action_history):
            return self.action_history[t]
        else:
            return None
    
    def reward_at(self, t):
        if 1 <= t < len(self.reward_history):
            return self.reward_history[t]
        else:
            return None