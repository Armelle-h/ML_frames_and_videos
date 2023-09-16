import os as os
import numpy as np

# torch:
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

# video processing:
import cv2

import sys
from PIL import Image
sys.path.insert(0,'/content/drive/MyDrive/ML_project_frames_and_videos/CLIP') # Delete if clip is installed by pip
import clip

class Pair_Frames(Dataset):
    """Input of init : video_dir, train=0, time_interval=2.0, length=100000, pos_ratio=0.5, seed=42, device='cpu'  
    """
    
    def __init__(self, video_dir, train=0, time_interval=2.0, length=100000, pos_ratio=0.5, seed=42, device='cpu'): # pas changer les paramètres par défault sans accord
        self.video_dir = video_dir #localisation of the videos
        self.train=train #0 for train, 1 for validation, 2 for test, the seed entered must be the same for the three
        self.seed = seed # seed for the sample determination (< 2**32-1-length)

        self.video_names = self.separation() #name of the videos to consider
        self.interval = time_interval #time interval between two frames of the same video in sec
        self.length = length #number of samples in the dataset (< 2**32-1)
        self.pos_ratio = pos_ratio # positive ratio
        
        self.device =device #select the device to do clip

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device) #create clip model
        

    def __len__(self):
        return self.length #number of samples in the dataset
  
    def __getitem__(self, idx): #loads and returns a sample from the dataset at the given index 
        name=f'{self.train}_{self.interval}_{self.length}_{self.pos_ratio}_{self.seed}_tensor{idx}.pt'
        #In the two following lines, for future uses replace with the path to the folder containing the saved features
        if os.path.exists('/content/drive/MyDrive/ML_project_frames_and_videos/tensor_features/'+name): #if there exists a file "tensoridx.pt"
        #in the folder "tensor_features"
          [image1, image2, label]=torch.load(os.path.join('/content/drive/MyDrive/ML_project_frames_and_videos/tensor_features', name), map_location=self.device) #making sure
          #it is downloaded under same device as everything, else problems with serialization 
          #reads from the file the values for image1, image2 and label

        else : #if we don't have a file yet i.e. if we never computed this index before, create it
        
          np.random.seed(idx + self.seed)
          label=np.random.binomial(1,self.pos_ratio)
          ### different videos
          if label==0: 
            #index of the videos
            same=True
            while(same): #don't want to have the same video
              i1=np.random.randint(0,len(self.video_names)) #choose two different random videos
              i2=np.random.randint(0,len(self.video_names))
              same= i1==i2
        
            #First video
            video_path = os.path.join(self.video_dir,self.video_names[i1]) #path of the video
            vid1=cv2.VideoCapture(video_path) #open the video
            f1=np.random.randint(1,vid1.get(cv2.CAP_PROP_FRAME_COUNT)) #index frame (1 because consider the position of video at frame f1-1 after, not so important not consider the very first frame)
            vid1.set(cv2.CAP_PROP_FRAME_COUNT,f1) #consider the position of the video at frame f1-1
            is_read, image1=vid1.read() #capture frame f1
            vid1.release() #release the ressources

            #Second video
            video_path = os.path.join(self.video_dir,self.video_names[i2]) #path of the video
            vid2=cv2.VideoCapture(video_path) #open the video
            f2=np.random.randint(1,vid2.get(cv2.CAP_PROP_FRAME_COUNT)) #index frame 
            vid2.set(cv2.CAP_PROP_FRAME_COUNT,f2) #consider the position of the video at frame f2-1
            is_read, image2=vid2.read() #capture frame f2
            vid2.release() #release the ressources


          ### same video
          else:
            i1=np.random.randint(0,len(self.video_names))#index of the video 
            video_path = os.path.join(self.video_dir,self.video_names[i1]) #path of the video
            vid1=cv2.VideoCapture(video_path) #open the video
            frame_interval=int(self.interval*vid1.get(cv2.CAP_PROP_FPS)) #number of frames in the chosen time interval
          
            #get the first frame
            f1=np.random.randint(1,vid1.get(cv2.CAP_PROP_FRAME_COUNT)-frame_interval-1) #index first frame # -1 juste pour être sûr
            vid1.set(cv2.CAP_PROP_FRAME_COUNT,f1) #consider the position of the video at frame f1-1
            is_read, image1=vid1.read() #capture frame f1

            #get the second frame
            f2=f1+frame_interval #use the interval to define the number of the second frame
            vid1.set(cv2.CAP_PROP_FRAME_COUNT,f2) #consider the position of the video at frame f2-1
            is_read, image2=vid1.read() #read the frame
            vid1.release() #close the video

          #Transform
        
          image1=self.clip_process(image1)
          image2=self.clip_process(image2)
          torch.save([image1, image2, label], os.path.join('/content/drive/MyDrive/ML_project_frames_and_videos/tensor_features', name))
          #For future uses, replace this with the path of the folder containing the saved features
          #we save in the folder "tensor_features" under the name "tensoridx" the values of image1, image2 and label associated to idx
        return image1, image2, label 

    def clip_process(self,image):
      """Transform an image to a vector of feature using CLIP and the model specified in the attributes of the class
      Args:image in form of a np.array
      Return:feature vector of image obtain with model by clip """
      
      image=Image.fromarray(np.uint8(image)).convert('RGB') #convert the image to the format used by CLIP
      image = self.preprocess(image).unsqueeze(0).to(self.device) #preprocess the image for clip and move it to the right device
      with torch.no_grad(): #using the CLIP model , find the important features of the image
        image_features = self.model.encode_image(image)
      return image_features.squeeze() # squeeze the extra dimension given by the output of clip

    def separation(self):
      """Create a disjoint partition of the videos into a train, validation and test set 
      and return the one corresponding by the task entered by the user in the attribute. 
      The seed must be the same when the dataset are constructed for the differents tasks. """
      np.random.seed(self.seed)
      video_names = np.array(os.listdir(self.video_dir))
      N = len(video_names)
      #shuffle the indices
      indices = np.arange(N)
      np.random.shuffle(indices)
      #define the split ratio
      split_ratio=[0.7,0.15,0.15]
      i = int(N * split_ratio[0])
      j = int(N * (split_ratio[0] +  split_ratio[1]))
      #create the 3 sets
      if self.train==0:
        return video_names[indices[0:i]] #train
      elif self.train==1:
        return video_names[indices[i:j]] #validation
      elif self.train==2:
        return video_names[indices[j:]] #test
      
