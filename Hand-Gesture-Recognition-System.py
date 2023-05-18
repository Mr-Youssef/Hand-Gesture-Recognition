#!/usr/bin/env python
# coding: utf-8

# ### **All Imports**

# In[11]:


import os
import time
from PIL import Image, ImageFilter
from Preprocessing_Utils import preprocessing_utils
from Feature_Extraction_Utils import feature_extraction_utils
from Classification_Utils import classification_utils
from pickle import load


# ### **Importing Data**

# In[12]:


directory = 'data'
images_paths = []
corrupted=0
for filename in os.listdir(directory):
    img = os.path.join(directory, filename)

    if not (img.endswith(".JPG") or img.endswith(".jpg")) and not (img.endswith(".png") or img.endswith(".PNG")):          # in case of desktop.ini file or any other file
          continue

    try:
        temp = Image.open(img)
    except (IOError, SyntaxError) as e:   # in case of corrupted images
        print('Bad file:', e)
        corrupted+=1
        continue 
    images_paths.append(img) 
print("Total number of images = ", len(images_paths))
print("Number of corrupted images  = ", corrupted)
images_paths.sort(key=lambda name: int(name.split('/')[-1].split('\\')[-1].split('.')[0]) ) # sort images according to their names


# ### **Preprocessing, Feature Extraction, and Predictions**

# In[17]:


mdl = load(open('Models/SVC_Model94.pkl', 'rb'))

predictions = []
elapsed_time = []
for img_path in images_paths:
    start = time.time()

    resized, gray, norm = preprocessing_utils.image_preprocessing(img_path)
    segmented_img = preprocessing_utils.image_segmentation(resized)
    noise_removal = preprocessing_utils.morphological_operations(segmented_img)
    edges = preprocessing_utils.canny_edge_detection(noise_removal)
    features = feature_extraction_utils.EOH(edges)
    y_pred = mdl.predict([features])
    
    end = time.time()

    predictions.append(y_pred)
    elapsed_time.append(end - start)


# ### **Output Files**

# In[18]:


file1 = open("results.txt","w")
for i in range(len(predictions)):
    file1.write(predictions[i][0] + "\n")
file1.close()


# In[19]:


file2 = open("time.txt","w")
for i in range(len(elapsed_time)):
    file2.write(str(round(elapsed_time[i], 3)) + "\n")
file2.close()


# ### **Generating Python Scripts**

# In[20]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script Hand-Gesture-Recognition-System.ipynb')


# In[21]:


if __name__ == '__main__':
    create_py()

