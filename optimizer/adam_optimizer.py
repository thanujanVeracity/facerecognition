
from .base_optimizer import BaseOptimizer
import torch.optim as optim
import yaml




class AdamOptimizer(BaseOptimizer):
    def __init__(self, config_file):
        super().__init__(config_file)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.betas = config['adam_optimizer']['betas']
        self.eps = float(config['adam_optimizer']['eps'])
        
    
    def get_optimizer(self, params):
        return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, eps=self.eps)
