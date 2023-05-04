import yaml

class BaseOptimizer:
    def __init__(self, config_file):
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.lr = config['base_optimizer']['learning_rate']
        self.weight_decay = config['base_optimizer']['weight_decay']
    
    def get_optimizer(self, params):
        return NotImplementedError


        
        