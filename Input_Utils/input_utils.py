#!/usr/bin/env python
# coding: utf-8

# #### **All Imports**

# In[ ]:


get_ipython().system('pip install gdown')


# In[6]:


import gdown
import zipfile
import os
from PIL import Image, ImageFilter
import copy
import random
from sklearn.model_selection import train_test_split
import random


# #### **Importing data from google drive**

# In[7]:


def read_from_drive():
    gdown.download('https://drive.google.com/uc?id=1JLxhdIddq6_vKlHml7jT48VaeXoJjvpR', 'dataset.zip', quiet=False)

    zip_ref = zipfile.ZipFile("/content/dataset.zip", 'r')
    zip_ref.extractall("dataset")
    zip_ref.close()


# ### **Importing data from a folder on your computer**

# In[8]:


def read_from_folder(folder_path):
    return folder_path


# #### Making sure all images have the same extension

# In[9]:


def same_extension():
    dataset_path = '/content/dataset'
    # Loop over the women and men folders
    for gender in ['Women', 'men']:
        gender_path = os.path.join(dataset_path, gender)
        for i in range(6):
            folder_path = os.path.join(gender_path, str(i))
            # Loop over the files in the folder
            extensions = set()
            for filename in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, filename)):
                    file_extension = os.path.splitext(filename)[1]
                    if file_extension not in ['.JPG']:
                        print(f'Error: {filename} has an invalid extension ({file_extension})')
                        os.remove(os.path.join(folder_path, filename))
                    extensions.add(file_extension)
            # Check that all files have the same extension
            if (len(extensions) == 0):
                print('All files have the same extension')


# #### Gathering all classes

# In[1]:


def gather_data(folder_path):    
  women_dataset = folder_path + '/Women'
  men_dataset = folder_path +'/men'

  images_paths = []
  # dataset_labels = []
  classes_sizes=[]
  corrupted=0
  corrupted_imgs=[]
  for i in range(6):
    women_folder_path = os.path.join(women_dataset, str(i))
    men_folder_path = os.path.join(men_dataset, str(i))
    # Get the file paths in the folders
    women_files = [os.path.join(women_folder_path, f) for f in os.listdir(women_folder_path) if os.path.isfile(os.path.join(women_folder_path, f))]
    men_files = [os.path.join(men_folder_path, f) for f in os.listdir(men_folder_path) if os.path.isfile(os.path.join(men_folder_path, f))]

    imgs = women_files + men_files
    for img in imgs:
      if not (img.endswith(".JPG") or img.endswith(".jpg")) and not (img.endswith(".png") or img.endswith(".PNG")):         # in case of desktop.ini file or any other file
          continue

      try:
          temp = Image.open(img)
      except (IOError, SyntaxError) as e:   # in case of corrupted images
          print('Bad file:', temp)
          corrupted+=1
          continue 
      path = img.split('/')
      images_paths.append(img)
      
      
    classes_sizes.append(len(imgs))    
  print("Total number of images = ", len(images_paths))
  print("Number of corrupted images  = ", corrupted)
  print("Number of images per class = ", classes_sizes)
  return images_paths


# In[11]:


def shuffle_and_get_labels(images_paths, input_type):
  random.seed(7)
  random.shuffle(images_paths)
  dataset_labels = []
  for img in images_paths:
    if input_type == 'drive':
      path=img.split('/')
      dataset_labels.append(path[4])
    elif input_type == 'folder':
      path=img.split('/')[-1].split('\\')
      dataset_labels.append(path[1])
  return images_paths, dataset_labels


# #### Splitting dataset into train and validation and test  (70%, 10%, 20%)

# In[1]:


def splitting(images_paths, dataset_labels):

    # splitting training into 80% and testing into 20%
    train_paths, test_paths, train_labels, test_labels = train_test_split(images_paths, dataset_labels, test_size=0.2, random_state=42, stratify=dataset_labels)

    # splitting training into 70% and validation into 10%
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

    print(f"Size of training data = {len(train_paths)}, Size of validation data = {len(validation_paths)}, Size of testing data = {len(test_paths)}")

    # check
    total = len(train_paths) + len(validation_paths) + len(test_paths)
    if total == len(images_paths):
        print("Splitting is done correctly")
    else:  
        print("Error in splitting")
    
    return train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels


# In[20]:


def input_data(input_type, folder_path = '/content/dataset'):       #default folder path if importing from drive
    if input_type == 'drive':
        read_from_drive()
        same_extension()
    elif input_type == 'folder':
        pass 
        
    images_paths = gather_data(folder_path)
    images_paths, dataset_labels = shuffle_and_get_labels(images_paths, input_type)
    train_images, train_labels, validation_images, validation_labels, test_paths, test_labels = splitting(images_paths, dataset_labels)
    
    return train_images, train_labels, validation_images, validation_labels, test_paths, test_labels


# In[2]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script input_utils.ipynb')


# In[3]:


if __name__ == '__main__':
    create_py()


# In[ ]:




