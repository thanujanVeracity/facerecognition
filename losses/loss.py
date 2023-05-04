"""This code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/utils.py
"""

import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance

class Loss(Function):
    """
    Abstract base class for distance measures.
    """
    def __init__(self, margin):
        super(Loss, self).__init__()
    
    def forward(self, anchor, positive, negative):
        """
        Compute the loss of the possitve and negative embeddings

        Args:
            anchor (np.ndarray): embedding of the anchor image
            positive (np.ndarray): embedding of the possitive image
            negative (np.ndarray): embedding of the negative image

        Returns:
            loss(float) : loss calculated from the embeddings
        """
        
        pass
        

class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dist, neg_dist):

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss
