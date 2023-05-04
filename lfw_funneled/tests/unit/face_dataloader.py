
# Install impportant pakages
import pytest
import requests
import torch
# from face_dataset import FaceDataset
import tempfile
import os
import shutil

import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#import units
# from file_details import write_file_details_to_csv
# from dataLoader import generate_csv_file
# from dataLoader.face_dataset import TripletFaceDatset
from dataLoader.face_dataloader import FaceDataLoader

# from face_dataset import TripletFaceDatset

import os
import shutil
import tempfile
import pytest
from torchvision import transforms
from unittest.mock import MagicMock

# to make temporary Image data
from PIL import Image  
width = 10
height = 10


def write_temporary_image_file( dir:str , name:str) -> str:
    # Create a temporary image file
    # Write the image data to the temporary file
    img = Image.new('RGB', (width, height))
    img.save(os.path.join(dir,name))

if(not os.path.isdir("./temp")):
    os.mkdir("./temp")
temp_dir = "./temp"
    
@pytest.fixture(scope="module")
def dataloader():
    # Make a temprory structured dataset
    
    dataset_dir = os.path.join(temp_dir, 'test_dataset')
    os.makedirs(dataset_dir)

    # Create a subdirectory for each person
    person1_dir = os.path.join(dataset_dir, 'elon')
    os.makedirs(person1_dir)
    person1 = write_temporary_image_file(person1_dir ,"person1.jpg")
    person1_1 = write_temporary_image_file(person1_dir, "person_1_1.jpg")

    person2_dir = os.path.join(dataset_dir, 'mark')
    os.makedirs(person2_dir)
    person2 = write_temporary_image_file(person2_dir, "person_2.jpg")
    person2_1 = write_temporary_image_file(person2_dir, "person_2_1.jpg")
    
    assert os.path.isdir(dataset_dir) == True
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

    dataloader = FaceDataLoader( dataset_dir, 6, 2, 2,None, 0.8,0.1, transform)
    
    return dataloader

def test_train_dataloader(dataloader):
    for i in dataloader.get_train_loader():
        assert list(i.keys())== ['anc_img', 'pos_img', 'neg_img', 'pos_class', 'neg_class']
        return
def test_valid_dataloader(dataloader):
    for i in dataloader.get_valid_loader():
        assert list(i.keys())== ['anc_img', 'pos_img', 'neg_img', 'pos_class', 'neg_class']
        return
def test_test_dataloader(dataloader):
    for i in dataloader.get_test_loader():
        assert list(i.keys())== ['anc_img', 'pos_img', 'neg_img', 'pos_class', 'neg_class']
        return
from models.mobilenetv2 import MobileNetV2Triplet
from optimizer.adam_optimizer import AdamOptimizer
from distance_measure import DistanceMeasure, L2Distance
from losses.loss import TripletLoss
from train import Trainer
def test_trainer(dataloader):
    trainer = Trainer(backbone= MobileNetV2Triplet, loss= TripletLoss, margin=0.1, optimizer=AdamOptimizer, distance_measure= L2Distance, device="cpu" ,log_dir=".")
    
    trainer.train(trainLoader=dataloader.get_train_loader(), valLoader=None, device="cpu", epochs=3, )
    
    
    
# Delete the temporary directory when you are done with it
shutil.rmtree(temp_dir)
