import numpy as np
from torch.nn.modules.distance import PairwiseDistance 
from abc import ABC, abstractmethod

class DistanceMeasure(ABC):
    """
    Abstract base class for distance measures.
    """
    
    @abstractmethod
    def __call__(self, a, b):
        """
        Compute the distance between two vectors a and b

        Args:
            a (numpy.ndarray): The first vector
            b (numpy.ndarray): The second vector
            
        Returns:
            float: The distance between the two vectors
        """
        pass
    
class EuclideanDistance(DistanceMeasure):
    """
    Concrete class implementing distance between two vectors a and b.
    """
    
    def __call__(self, a, b):   
        """
        Compute the Euclidean distance between two vectors a and b.
        
            Parameters:
                a (numpy.ndarray): The first vector.
                b (numbpy.ndarray): The second vector.
                
            Returns:
                float : The Euclidean distance between the two vectors.
        """
    
        return np.linalg.norm(a-b)
   
   
    
class ManhattanDistance(DistanceMeasure):
    """
    Concrete class implementing distance between two vectors a and b.
    """
    
    def __call__(self, a, b):   
        """
        Compute the Manhattan distance between two vectors a and b.
        
        Parameters:
            a (numpy.ndarray): The first vector.
            b (numbpy.ndarray): The second vector.
            
        Returns:
            float : The Euclidean distance between the two vectors.
        """
    
        return np.sum(np.abs(a-b))
    
    
class L2Distance(DistanceMeasure):
    """
    Concrete class implementing distance between two vectors a and b.
    """
    
    def __call__(self, a, b):   
        """
        Compute the Pairwise distance between two vectors a and b.
        
        Parameters:
            a (numpy.ndarray): The first vector.
            b (numbpy.ndarray): The second vector.
            
        Returns:
            float : The Pairwise distance between the two vectors.
        """
    
        return PairwiseDistance(p=2).forward(a, b)
    
    

