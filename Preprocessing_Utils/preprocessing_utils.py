#!/usr/bin/env python
# coding: utf-8

# ### All imports

# In[ ]:


import numpy as np
import cv2 as cv


# ### Image preprocessing

# ##### resizing -> grayscale -> normalization

# In[ ]:


def image_preprocessing(img_path):
    target_size = (200, 200)
    # Load the image
    image = cv.imread(img_path)

    # Resize the image to the target size
    resized_img = cv.resize(image, target_size)

    # Convert the resized image to grayscale
    gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

    # Normalize the pixel values to be between 0 and 1
    # normalized_img  = gray_img / 255.0
    normalized_img = (gray_img - np.min(gray_img)) * 255.0 / (np.max(gray_img) - np.min(gray_img))

    return resized_img, gray_img, normalized_img    # return the original and the preprocessed images


# #### Image Segmentation

# In[ ]:


def image_segmentation(img):
    # Apply Gaussian blur to remove noise (optional)
    img = cv.GaussianBlur(img, (5, 5), 0)

    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    ret,thresh = cv.threshold(skinRegionHSV,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    return thresh


# ### Morphological Operations

# In[ ]:


def morphological_operations(img):
    # Define the kernel for morphological operations
    kernel = np.ones((5,5),np.uint8)

    # Perform dilation operation on the image
    dilated_img = cv.dilate(img, kernel, iterations=1)

    # Perform erosion operation on the image
    eroded_img = cv.erode(img, kernel, iterations=1)

    # Perform opening operation on the image
    opening_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    # Perform closing operation on the image
    closing_img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    return closing_img


# ### Canny Edge Detection

# In[ ]:


def canny_edge_detection(img):
    # Apply Gaussian blur to reduce noise in the image
    img_blur = cv.GaussianBlur(img, (5,5), 0)

    # Perform Canny edge detection
    edges = cv.Canny(img_blur, 100, 200)

    return edges


# In[ ]:


def preprocessing(images_paths):
    imgs_edges = []
    imgs_gray = []
    for img in images_paths:
        resized, gray, norm = image_preprocessing(img)
        segmented_img = image_segmentation(resized)
        noise_removal = morphological_operations(segmented_img)
        edges = canny_edge_detection(noise_removal)
        imgs_gray.append(gray)
        imgs_edges.append(edges)
    return imgs_edges, imgs_gray


# In[1]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script preprocessing_utils.ipynb')


# In[2]:


if __name__ == '__main__':
    create_py()


# In[ ]:




