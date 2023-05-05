#import the packages
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
# from dataLoader.face_dataset import TripletFaceDatset
from distance_measure.distance_measure import DistanceMeasure, L2Distance

from losses.loss import Loss
from torch.utils.data import DataLoader
from optimizer.base_optimizer import BaseOptimizer
from torch.utils.data import Dataset
from losses.loss import TripletLoss

class Trainer:
    def __init__(self,  train_iterator: Dataset, valid_iterator: Dataset ,model: nn.Module, loss: Loss, margin: float, optimizer: BaseOptimizer, device, num_workers=1, log_dir=None):
        
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.loss = loss(margin)
        self.model = model
        self.optimizer = optimizer("./optimizer/optimizer_config.yaml").get_optimizer(params=self.model.parameters())
        self.device = device
    
    def _train_epoch(self, epoch, batch_size, num_workers):
        # Set the model to train mode(this is the default)
        self.model.train()
        
        #valid data should be checked from the unseen data directory
        
        # for each epoch new set of triplet should be generated
        # I can randomly choose data from the face dataset and generate a new set of triplet
        # But for the validation set i cannot generate a new set of tripletes from the same directory 
        # because it should be from unseen data directory
        # So I need to generate a new set of triplets from the unseen data directory for the validation.
        
        
        train_dataloader = DataLoader(dataset = next(self.train_iterator),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=False)
        
        losses = []
        num_valid_training_triplets = 0
        
        # Loop through the batches
        for batch, (sample) in enumerate(tqdm(train_dataloader)):
            

            # Take anchor possitve and negative images
            ancImages = sample["anc_img"].to(self.device)
            posImages = sample["pos_img"].to(self.device)
            negImages = sample["neg_img"].to(self.device)


            #! changed
            # Concatenate the input images into one tensor because doing multiple forward passes would create
            # weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
            # issue
            # allImages = torch.cat((ancImages, posImages, negImages))

            #! batch size
            # Do a foraward pass with all images and get the embedding for the images
            pos_dist, neg_dist = self.model(ancImages, posImages, negImages)
            
            # Calculate the triplet loss
            loss = self.loss.forward(pos_dist, neg_dist)
            losses.append(loss)
            
            # Calculating number of triplets that met the triplet selection method during the epoch
            num_valid_training_triplets += len(pos_dist)

            
            #Zero the optimier gradients (they accumulate by default)
            self.optimizer.zero_grad()

            # perform Backpropagation
            loss.backward()

            # Step the optimizer (gradient decent)
            self.optimizer.step()
        
         # Print training statistics for epoch and add to log
        print('Epoch {}:\tNumber of valid training triplets in epoch: {}\t AverageLoss {}'.format(
                epoch,
                num_valid_training_triplets,
                torch.mean(losses)
            )
        )
            
    def _valid_epoch(self, epoch, batch_size, num_workers):
        # Set the model to evaluate mode
        self.model.eval()
        metric = []
        losses = []
        num_valid_validating_triplets = 0
        
        valid_dataloader = DataLoader(dataset = next(self.valid_iterator),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=False)
        
        


        # Loop through the batches
        for batch, (sample) in enumerate(tqdm(valid_dataloader)):
            

            # Take anchor possitve and negative images
            ancImages = sample["anc_img"].to(self.device)
            posImages = sample["pos_img"].to(self.device)
            negImages = sample["neg_img"].to(self.device)

            #! batch size
            # Do a foraward pass with all images and get the embedding for the images
            pos_dist, neg_dist = self.model(ancImages, posImages, negImages)
            
            # Calculate the triplet loss
            loss = self.loss.forward(pos_dist, neg_dist)
            
            losses.append(loss)
            num_valid_validating_triplets += len(pos_dist)
            # metric.append()
         # Print training statistics for epoch and add to log
        print('Epoch {}:\tNumber of valid training triplets in epoch: {}\t AverageLoss {}'.format(
                epoch,
                num_valid_validating_triplets,
                torch.mean(losses)
            )
        )
            
        
    # Train time!
    def train(self ,epochs, batch_size, num_workers, validate_every):
        
        generate = True
        

        # Loop through the epochs
        for epoch in range(epochs):
            

            
            # Train the model for one epoch
            self._train_epoch(epoch=epoch, batch_size= batch_size, num_workers= num_workers)
            
            if (epoch % validate_every )and (self.valid_data != None) == 0:
                
                #Do evaluation on validation dataset
                self.model.eval()
                        
                with torch.no_grad():
                    self._valid_epoch(epoch=epoch, batch_size= batch_size, num_workers= num_workers)   
                    

            
        
           

            return
        


