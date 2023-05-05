
import torch
import torch.nn as nn
from distance_measure.distance_measure import DistanceMeasure, L2Distance
import numpy as np
from models.mobilenetv2 import MobileNetV2Triplet

class SiameseNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, distance_measure: DistanceMeasure, margin: float, useSemihardNegatives: bool = False):
        # Ensures that the parent class is properly initialized 
        # before any additional initialization steps in the child 
        # class are performed
        super(SiameseNetwork, self).__init__()
        
        self.backbone = backbone()
        self.distance_measure = distance_measure()
        self.useSemihardNegatives = useSemihardNegatives
        self.margin = margin
        
        
    def forward(self, x1, x2, x3):
        
        x1_len = x1.shape[0]
        x2_len = x2.shape[0]
        x3_len = x3.shape[0]
        
        embs = self.backbone(torch.cat((x1,x2,x3)))
        
        x1_emp = embs[:x1_len]
        x2_emp = embs[x1_len:x1_len+x2_len]
        x3_emp = embs[x1_len+x2_len:]
        
        #calculate the possitve distance
        pos_dist = self.distance_measure(x1_emp, x2_emp)
        
        #calculate the negative distance        
        neg_dist = self.distance_measure(x1_emp, x3_emp)

        # Sample mining
        # validate  whether the possitive distance is smaller than distance
        if self.useSemihardNegatives:
            # Semi-Hard Negative triplet selection
            #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
            #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
            first_condition = (neg_dist - pos_dist < self.margin).cpu().numpy().flatten()
            second_condition = (pos_dist < neg_dist).cpu().numpy().flatten()
            all = (np.logical_and(first_condition, second_condition))
            valid_triplets = np.where(all == 1)
        else:
            # Hard Negative triplet selection
            #  (negative_distance - positive_distance < margin)
            #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
            all = (neg_dist - pos_dist < self.margin).cpu().numpy().flatten()
            valid_triplets = np.where(all == 1)
            
        # The embeddings which is validate, are taken 
        x1_emp_valid = x1_emp[valid_triplets]
        x2_emp_valid = x2_emp[valid_triplets]
        x3_emp_valid = x3_emp[valid_triplets]
        
        pos_dist = self.distance_measure(x1_emp_valid, x2_emp_valid)
        neg_dist = self.distance_measure(x1_emp_valid, x3_emp_valid)
        
        print(pos_dist[0], neg_dist[0])
        
        
        return pos_dist, neg_dist
    
if __name__ == "__main__":
    
    model = SiameseNetwork(
        backbone= MobileNetV2Triplet, distance_measure=L2Distance, margin=0.1)
    
    x = torch.randn(2, 3, 2, 2)
    y = torch.randn(2, 3, 2, 2)
    z = torch.randn(2, 3, 2, 2)
    
    x_len = x.shape[0] 
    y_len = y.shape[0]   
    z_len = z.shape[0]   
    # print(torch.cat((x,y,z)))
    # print(torch.cat((x,y,z))[0]==x)
    
    print(x.shape[0], y.shape, z.shape)
    print(".........................")
    print(x)
    print(".........................")
    print(y)
    print(".........................")
    print(z)
    print(".........................")
    
    
    pos_dist, neg_dist = model.forward(x, y, z)
    
    print(pos_dist)
    print(neg_dist)
    
    
        
        
        
        
    