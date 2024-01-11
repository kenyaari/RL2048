from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    rng = np.random.default_rng()
    
    @staticmethod
    @abstractmethod
    def get_initial_state(): pass
    
    @staticmethod
    @abstractmethod
    def transition(state: int, action: int): pass
    
    @staticmethod
    @abstractmethod
    def get_valid_actions(state: int): pass
    
    @staticmethod
    @abstractmethod
    def get_all_next_states(state: int, action: int): pass
    
    @staticmethod
    @abstractmethod
    def reward(current_state: int, current_action: int, next_state: int): pass
    
    @staticmethod
    @abstractmethod
    def get_feature_vector(state: int, action: int): pass
    
    @staticmethod
    @abstractmethod
    def is_terminal_state(state: int): pass
    
    @staticmethod
    @abstractmethod
    def get_state_status(state: int): pass