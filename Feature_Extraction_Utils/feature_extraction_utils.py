#!/usr/bin/env python
# coding: utf-8

# ### All Imports

# In[ ]:


import numpy as np
import cv2 as cv
import pandas as pd


# ### Edge Of Oriented Histogram

# In[ ]:


def EOH(img):
    # Compute the gradient magnitude and direction using Sobel operators
    dx = cv.Sobel(img, cv.CV_32F, 1, 0)
    dy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, angle = cv.cartToPolar(dx, dy, angleInDegrees=True)

    # Define the number of histogram bins and range
    hist_bins = 9
    hist_range = (0, 180)

    # Compute the histogram of gradient orientations for each cell of a 4x4 grid
    cell_size = (img.shape[0] // 4, img.shape[1] // 4)
    hog_descriptor = np.zeros((4, 4, hist_bins))
    for i in range(4):
        for j in range(4):
            cell_mag = mag[i*cell_size[0]: (i+1)*cell_size[0], j*cell_size[1]: (j+1)*cell_size[1]]
            cell_angle = angle[i*cell_size[0]: (i+1)*cell_size[0], j*cell_size[1]: (j+1)*cell_size[1]]
            hist, _ = np.histogram(cell_angle, bins=hist_bins, range=hist_range, weights=cell_mag)
            hog_descriptor[i, j] = hist

    # Flatten the descriptor to obtain a feature vector for the entire image
    hog_descriptor = hog_descriptor.flatten()

    # Normalize the feature vector using L2 normalization
    hog_descriptor /= np.linalg.norm(hog_descriptor)

    return hog_descriptor


# In[ ]:


def saving_csv(features, labels, name):
    features_pd = pd.Series(features)
    labels_pd = pd.Series(labels)
    data_df = pd.concat([features_pd, labels_pd], axis=1)  
    data_df.columns = ['features', 'labels']
    data_df.to_csv(f'Datasets/TwoFeatures/{name}.csv', index=False)


# In[ ]:


def feature_extraction(imgs_edges, labels, name):
    all_features = []
    for edge in imgs_edges:
        features= EOH(edge)
        all_features.append(features)
    saving_csv(all_features, labels, name)
    return all_features


# In[6]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script feature_extraction_utils.ipynb')


# In[7]:


if __name__ == '__main__':
    create_py()


# In[ ]:




