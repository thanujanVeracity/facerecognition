from base_optimizer import BaseOptimizer
import torch.optim as optim
import yaml

class SGDOptimizer(BaseOptimizer):
    def __init__(self, config_file):
        super().__init__(config_file)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.momentum = config['sgd_optimizer']['momentum']
    
    def get_optimizer(self, params):
        return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
