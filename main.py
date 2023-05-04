import os
from torchvision import transforms
from optimizer.adam_optimizer import AdamOptimizer
from train import Trainer
from dataLoader.face_dataloader import FaceDataLoader
from dataLoader.face_dataset import TripletFaceDatset
from losses.loss import TripletLoss
from distance_measure.distance_measure import L2Distance
from models.mobilenetv2 import MobileNetV2Triplet
import argparse


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    
    
    
    # # number of tripletes to generate for each epoch
    # parser.add_argument('--num_triplets', type=int, default=)
    # # size of the each triplet batch
    # parser.add_argument('--triplet_batch_size', type=int, default=544)
    # # number of human classes should be in one batch
    # parser.add_argument('--num_human_identities_per_batch', type=int, default=32)
    # # number of epochs
    # parser.add_argument('--epoch', type=int, default=3)
    # # file of triplet file
    # parser.add_argument('--triplet_file', type=str, default=None)
    # # transoform function
    # parser.add_argument('--transform', type=str, default=None)
    
    # args = parser.parse_args()
    
    
    
    # First thing we need data to train the model. Therefore the Dataset is created from the root directory.
    # Since the dataloader expects the data in a format and our data in the folder in different format, we need
    # to use adapter design pattern to create the dataset.
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    
    # transform function to transform the dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    # dataset = TripletFaceDatset( dataset_dir, 6, 2, 2,3)
    
    # Then trainer is called to train the model. But here Trainer class is calling the Dataset  inside, which brokes the SOLID rule
    # dataloader = FaceDataLoader( dataset_dir, 6, 2, 2,None, 0.8,0.1, transform)
    trainer = Trainer(train_root = "train", valid_root = "valid", transform= transform, step_per_epoch = 100 , batch_size = 32, num_triplets = 3200 , backbone= MobileNetV2Triplet, loss= TripletLoss, margin=0.1, optimizer=AdamOptimizer, distance_measure= L2Distance, device="cpu" ,log_dir=".")
    
    #Then i am training the model.
    trainer.train(epochs= 2, validate_every=1)
    
    