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

from losses.loss import TripletLoss

class Trainer:
    def __init__(self, train_root: str, valid_root: str, transform, step_per_epoch: int, batch_size: int, num_triplets:int, backbone: nn.Module, loss: Loss, margin: float, optimizer: BaseOptimizer, distance_measure: DistanceMeasure, device, num_workers=1, log_dir=None):
        

        self.train_root: str = train_root
        self.valid_root: str = valid_root
        self.steps_per_epoch: int = step_per_epoch
        self.batch_size: int = batch_size
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        
        
        
        self.distance_measure = distance_measure
        self.loss = loss(margin)
        self.model = SiameseNetwork( backbone= backbone, distance_measure = self.distance_measure, margin=margin).to(device=device)
        self.optimizer = optimizer("/home_2/thajan/Desktop/faceregognition/mine/optimizer/optimizer_config.yaml").get_optimizer(params=self.model.parameters())
        self.device = device
        
    def _train_epoch(self, epoch, generate):
        # Set the model to train mode(this is the default)
        self.model.train()
        
        #valid data should be checked from the unseen data directory
        
        # for each epcoh new set of dataset should be generated
        train_dataloader = DataLoader(dataset = TripletFaceDatset(
            root_dir= self.train_root,
            csv_name=os.path.join(self.train_root, "train.csv"),
            num_triplets=self.steps_per_epoch * self.batch_size,
            num_human_identities_per_batch=self.num_human_identities_per_batch,
            triplet_batch_size=self.batch_size,
            transform=self.transform,
            epoch= epoch,
            generate = generate,
            ),
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
        
        valid_dataloader = DataLoader(dataset = TripletFaceDatset(
            root_dir= self.valid_root,
            num_triplets=self.steps_per_epoch * self.batch_size,
            num_human_identities_per_batch=self.num_human_identities_per_batch,
            triplet_batch_size=self.batch_size,
            transform=self.transform,
            epoch=epoch,
            generate = generate,
            ),
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
            self._train_epoch(epoch=epoch,generate=generate)
            
            if epoch % validate_every == 0:
                
                #Do evaluation on validation dataset
                self.model.eval()
                        
                with torch.no_grad():
                    self._valid_epoch(epoch=epoch,generate=generate)   
                    
            if(generate == True):
                generate = False

            
        
            # Print training statistics for epoch and add to log
            print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(
                    epoch,
                    num_valid_training_triplets
                )
            )

            return
        


