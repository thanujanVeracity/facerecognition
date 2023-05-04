#import the packages
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from dataLoader.face_dataset import TripletFaceDatset
from distance_measure.distance_measure import DistanceMeasure, L2Distance
from SiameseNetwork import SiameseNetwork
from losses.loss import Loss
from torch.utils.data import DataLoader
from optimizer.base_optimizer import BaseOptimizer
from torch.utils.data import Dataset
from losses.loss import TripletLoss

class Trainer:
    def __init__(self,  train_data: Dataset, valid_data: Dataset, batch_size: int ,backbone: nn.Module, loss: Loss, margin: float, optimizer: BaseOptimizer, distance_measure: DistanceMeasure, device, num_workers=1, log_dir=None):
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.num_workers = num_workers
        self.distance_measure = distance_measure
        self.loss = loss(margin)
        self.model = SiameseNetwork( backbone= backbone, distance_measure = self.distance_measure, margin=margin).to(device=device)
        self.optimizer = optimizer("/home_2/thajan/Desktop/faceregognition/mine/optimizer/optimizer_config.yaml").get_optimizer(params=self.model.parameters())
        self.device = device
        
    def _train_epoch(self, epoch, generate):
        # Set the model to train mode(this is the default)
        self.model.train()
        
        #valid data should be checked from the unseen data directory
        
        # for each epoch new set of triplet should be generated
        # I can randomly choose data from the face dataset and generate a new set of triplet
        # But for the validation set i cannot generate a new set of tripletes from the same directory 
        # because it should be from unseen data directory
        # So I need to generate a new set of triplets from the unseen data directory for the validation.
        
        
        train_dataloader = DataLoader(dataset = self.train_data,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False)
        

        num_valid_training_triplets = 0
        
        # Loop through the batches
        for batch, (sample) in enumerate(tqdm(train_dataloader)):
            
            
            
            print("Training on batch {}".format(batch))
            
            

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
            
            # Calculating number of triplets that met the triplet selection method during the epoch
            num_valid_training_triplets += len(pos_dist)
            

            
            #Zero the optimier gradients (they accumulate by default)
            self.optimizer.zero_grad()

            # perform Backpropagation
            loss.backward()

            # Step the optimizer (gradient decent)
            self.optimizer.step()
            
    def _valid_epoch(self, epoch, generate):
        # Set the model to evaluate mode
        self.model.eval()
        metric = []
        losses = []
        
        valid_dataloader = DataLoader(dataset = self.valid_data,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
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
            # metric.append()
            
        
    # Train time!
    def train(self ,epochs, validate_every):
        
        generate = True
        

        # Loop through the epochs
        for epoch in range(epochs):
            

            num_valid_training_triplets = 0
            
            
            # Train the model for one epoch
            self._train_epoch(epoch=epoch)
            
            if (epoch % validate_every )and (self.valid_data != None) == 0:
                
                #Do evaluation on validation dataset
                self.model.eval()
                        
                with torch.no_grad():
                    self._valid_epoch(epoch=epoch)   
                    

            
        
            # Print training statistics for epoch and add to log
            print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(
                    epoch,
                    num_valid_training_triplets
                )
            )

            return
        


