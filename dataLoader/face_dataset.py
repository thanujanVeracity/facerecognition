"""The original code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/data_loader.py
    The code was modified to speed up the triplet generation process by bypassing the dataframe.loc operation,
    generate batches according to a set amount of human identities (classes) per triplet batch, and to
    support .png, .jpg, and .jpeg files.
"""



# Import all the necessary packages
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import glob
import time
import sys


class TripletsFaceDataset(Dataset):
    
    def __init__(self, triplets: list):
        """This class implements tr

        Args:
            triplets (list): _description_
        """
        
        self.triplets = triplets
        
        return
        
    def __len__(self):
        if self.training_triplets:
            return len(self.training_triplets)
        else:
            return 0
    
    # Added this method to allow .jpg, .png, and .jpeg image support
    def _add_extension(self, path):
        
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # print(os.listdir(path))
        
        if os.path.exists(path+'.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        elif os.path.exists(path + '.jpeg'):
            return path + '.jpeg'
        else:
            raise RuntimeError('No file "{}" with extension .png or .jpg or .jpeg'.format(path))
    
        return
 
    
    def __getitem__(self, idx):
        if self.training_triplets:
            

            anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

            anc_img = self._add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
            pos_img = self._add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
            neg_img = self._add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))

            # Modified to open as PIL image in the first place
            anc_img = Image.open(anc_img)
            pos_img = Image.open(pos_img)
            neg_img = Image.open(neg_img)

            pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
            neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

            sample = {
                'anc_img': anc_img,
                'pos_img': pos_img,
                'neg_img': neg_img,
                'pos_class': pos_class,
                'neg_class': neg_class
            }

            if self.transform:
                sample['anc_img'] = self.transform(sample['anc_img'])
                sample['pos_img'] = self.transform(sample['pos_img'])
                sample['neg_img'] = self.transform(sample['neg_img'])

            return sample
        else:
            None
        
        return sample   

class TripletFaceIterator:

    def __init__(self,
                root_dir: str, 
                csv_name: str,
                generate: bool,
                num_triplets, 
                epoch, 
                num_human_identities_per_batch=32,
                triplet_batch_size=544,
                triplets_file=None, 
                transform=None):
        
        """
        This code defines a custom dataset class TripletFaceDataset that generates triplets for triplet loss 
        based on a given directory of face images or from the face images csv file.
        
        |face_images
            |person1
                |image
                |image
            |person2
                |image
                |image
            |person3
                |image
                |image
                
                
        This class generate a new triplet dataset when each time it is called
        
        The class generates a dictionary that maps each class name to a list of image ids and uses this dictionary
        to randomly select anchor, positive, and negative images for each triplet. 
        
        Additionally, it can optionally load pre-generated triplets or a CSV file with metadata. 
        Finally, it has an option for applying image transformations during training.

        Args:
            root_dir (_type_): Absolute path to dataset.
            training_dataset_csv_path (_type_): _description_
            num_triplets (_type_):  Number of triplets required to be generated.
            epoch (_type_): Current epoch number (used for saving the generated triplet list for this epoch).
            num_human_identities_per_batch (int, optional):Number of set human identities per batch size. Defaults to 32.
            triplet_batch_size (int, optional): _description_. Defaults to 544.
            triplets_file (_type_, optional): Path to a pre-generated triplet numpy file to skip the triplet generation process (Only
                                    will be used for one epoch). Defaults to None.
            transform (_type_, optional): Required image transformation (augmentation) settings. Defaults to None.
        """        

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        # VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        # forcing the 'name' column as being of type 'int' instead of type 'object')

        
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.epoch = epoch
        self.transform = transform
        
        # Create a CSV file and dataframe that includes information about the face dataset and its corresponding images.
        if(not os.path.isfile(csv_name)):
            self.dir_detail_df = self._generate_csv_file(root_dir, csv_name=csv_name)
            
        # Load the dataset frame from the CSV file which contains the face dataset details
        else:
            print("Loading pre-generated csv file...")
            self.dir_detail_df = pd.read_csv(csv_name)

        # Modified here to bypass having to use pandas.dataframe.loc for retrieving the class name
        #  and using dataframe.iloc for creating the faceClasses dictionary
        df_dict = self.dir_detail_df.to_dict()
        self.df_dict_class_name = df_dict["name"]
        self.df_dict_id = df_dict["id"]
        self.df_dict_class_reversed = {value: key for (key, value) in df_dict["class"].items()}
        self.triplets_file = os.path.join(root_dir, "triplets.npy")
        
        # if generate or not os.path.exists(self.triplets_file):
        #     self.training_triplets = self._generate_triplets()
        # else:
        #     print("Loading pre-generated triplets file ...")
        #     self.training_triplets = np.load(triplets_file)

        return  
    
    
    def _makeDictionaryForFaceClass(self):
        """

        Returns:
            Dictionary : {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        
        faceClasses = dict()
        for idx, label in enumerate(self.dir_detail_df['class']):
            if label not in faceClasses:
                faceClasses[label] = []
            # Instead of utilizing the computationally intensive pandas.dataframe.iloc() operation
            faceClasses[label].append(self.df_dict_id[idx])

        return faceClasses

    def _generate_triplets(self):
        triplets = []

        classes = self.dir_detail_df['class'].unique()
        faceClasses = self._makeDictionaryForFaceClass()
        
        print("\nUnique classes {}".format(classes))
        print("\nFace classes {}".format(faceClasses))
        print("\nGenerating {} triplets ...".format(self.num_triplets))

        numTrainingIterationsPerProcess = self.num_triplets / self.triplet_batch_size
        print("Number of iterations{}".format(numTrainingIterationsPerProcess))
        
        for training_itr in tqdm(range(int(numTrainingIterationsPerProcess))):

            """
            For each batch:

                - Randomly choose set amount of human identities (classes) for each batch
                
                --notes -> here the strategy for generating datasets is they are making a
                mew back of triplets for each epoch training.

                - For triplet in batch:
                    - Randomly choose anchor, positive and negative images for triplet loss
                    - Anchor and possitive images in possitive class
                    - Negative image in negative class
                    - At least, two images needed for anchor and possitive images in possitve class
                    - Negative image should have different class as anchor and posstive images by definition
            """

            #Randomly choose a set of human identities
            classesPerBatch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)
            print("\nClasses per batch".format(classesPerBatch))


            # Traverse through the whole batch size
            for triplet in range(self.triplet_batch_size):
                print("generating triplet number {}".format(triplet))
                
                #randomly choose a possitive class and a negative calss
                pos_class = np.random.choice( classesPerBatch )
                neg_class = np.random.choice( classesPerBatch )

                # Make sure to find atleast 2 faces of that possitive class
                while len(faceClasses[pos_class]) < 2:
                    pos_class = np.random.choice( classesPerBatch )

                # Make sure to find the different human identies as possitve and negative classes
                while pos_class == neg_class:
                    neg_class = np.random.choice( classesPerBatch )


                # Instead of utilizing the computationally intensive pandas.datafrane.ioc() operation
                #find the index of possitive class and find the name of the possitve class
                pos_name_index = self.df_dict_class_reversed[pos_class]
                pos_name = self.df_dict_class_name[pos_name_index]

                #find the index of negative class and find the name of the negative class
                neg_name_index = self.df_dict_class_reversed[neg_class]
                neg_name = self.df_dict_class_name[neg_name_index]

                # if faces in one class is to take one as anchor and other as negative
                if len(faceClasses[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                # other-wise find different faces as anchor and possitive in one class
                else:
                    ianc = np.random.randint(0, len(faceClasses[pos_class]))
                    ipos = np.random.randint(0, len(faceClasses[pos_class]))

                    while ianc == ipos:
                        ipos = np.random.randint(0, len(faceClasses[pos_class]))
                
                # Now we can take a random id as a negative face
                ineg = np.random.randint(0, len(faceClasses[neg_class]))

                # Append the three faces and names and id of the possitive and negative classes
                triplets.append(
                    [
                        faceClasses[pos_class][ianc],
                        faceClasses[pos_class][ipos],
                        faceClasses[neg_class][ineg],
                        pos_class,
                        neg_class,
                        pos_name,
                        neg_name
                    ]
                )
        

        print("Saving training triplets list in 'datasets/generated_triplets' directory ...")
        np.save('datasets/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                self.epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ),triplets
        )

        print("Training triplets' list Saved!\n")

        return triplets
    
    def _generate_csv_file(self, dataroot, csv_name):
        """Generates a csv file containing the image paths of the glint360k dataset for use in triplet selection in
        triplet loss training.

        Args:
            dataroot (str): absolute path to the training dataset.
            csv_name (str): name of the resulting csv file.
        """
        print("\nLoading image paths ...")
        files = glob.glob(dataroot + "/*/*")
        
        
        

        start_time = time.time()
        list_rows = []

        print("Number of files: {}".format(len(files)))
        print("\nGenerating csv file ...")

        progress_bar = enumerate(tqdm(files))

        for file_index, file in progress_bar:

            face_id = os.path.basename(file).split('.')[0]
            face_label = os.path.basename(os.path.dirname(file))

            # Better alternative than dataframe.append()
            row = {'id': face_id, 'name': face_label}
            list_rows.append(row)
            
            

        dataframe = pd.DataFrame(list_rows)
        print(dataframe)
        dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)

        # Encode names as categorical classes
        dataframe['class'] = pd.factorize(dataframe['name'])[0]
        dataframe.to_csv(path_or_buf=csv_name, index=False)
        elapsed_time = time.time()-start_time
        print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


        return dataframe

    def __iter__(self):
      self.a = 1
      return self
  
    def __next__(self):
        if self.a == 1:
            self.training_triplets = self._generate_triplets()
            self.a = 0
            return TripletsFaceDataset(self.training_triplets)
        else:
            raise StopIteration
        
    # def get_dir_detail_df(self):
    #     return self.dir_detail_df
    
    #  def clean(self):
    #     try :
    #         self.root_dir = None
    #         self.num_triplets = None
    #         self.num_human_identities_per_batch = None
    #         self.triplet_batch_size = None
    #         self.epoch = None
    #         self.transform = None
    #         self.dir_detail_df = None
            
    #         self.df_dict_class_name = None
    #         self.df_dict_id = None
    #         self.df_dict_class_reversed = None
            
    #         files = glob.glob('datasets/generated_triplets/*')
    #         for f in files:
    #             os.remove(f)
        
        
    #         return True
        
    #     except:
    #         print("isse")
        
        
        








