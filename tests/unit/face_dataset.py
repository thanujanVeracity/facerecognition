
# Install impportant pakages
import pytest
import requests
import torch
# from face_dataset import FaceDataset
import tempfile
import os
import shutil
#import units
# from file_details import write_file_details_to_csv
from dataLoader import generate_csv_file
from dataLoader.face_dataset import TripletFaceDatset
from dataLoader.face_dataloader import FaceDataLoader

# from face_dataset import TripletFaceDatset

import os
import shutil
import tempfile
import pytest

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
    
print("--------------setup----------------")
    # Create a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    
    @pytest.fixture(scope="module")
    def dataset():
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
        
        dataset = TripletFaceDatset( dataset_dir, 6, 2, 2,3)
        
        return dataset

    def test_the_dataset_length_before_creating_triplets(dataset):
        assert len(dataset) == 6
        
    
    def test_check_path_existence(self, mock_isdir):
        mock_isdir.return_value = True
    
    def test_the_dataset_elements_before_creating_triplets(dataset):
        
        

        # mocker.patch("dataset._add_extension", return_value=)
        assert dataset.__getitem__(0) == None
        
    def test_get_dir_detail_df(dataset):
        assert dataset.get_dir_detail_df().to_dict() == { 'id': {0: 'person1', 1: 'person_1_1', 2: 'person_2', 3: 'person_2_1'}, 'name': {0: 'elon', 1: 'elon', 2: 'mark', 3: 'mark'}, 'class': {0: 0, 1: 0, 2: 1, 3: 1}}
        
    def test__generate_triplets(dataset):
        assert dataset._generate_triplets() == [['person_2_1', 'person_2', 'person1', 1, 0, 'mark', 'elon'], ['person_1_1', 'person1', 'person_2_1', 0, 1, 'elon', 'mark'], ['person1', 'person_1_1', 'person_2_1', 0, 1, 'elon', 'mark'], ['person1', 'person_1_1', 'person_2_1', 0, 1, 'elon', 'mark'], ['person1', 'person_1_1', 'person_2', 0, 1, 'elon', 'mark'], ['person_2', 'person_2_1', 'person_1_1', 1, 0, 'mark', 'elon']]
        
    
    def test_clean(dataset):
        assert dataset.clean()
    
    
        
        
        

