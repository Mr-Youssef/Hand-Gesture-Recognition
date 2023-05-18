---
authors:
    - "Rana Gamal (1190449)"
    - "Omar Alaa (1190377)"
    - "Mai Abdelhameed (1190365)"
    - "Youssef Mohamed Mahmoud (1190202)"

title: "Hand Gesture Recognition CMPN450 Project"
abstract: "This project was made using tools learned in the Pattern Recognition course and used [Static Hand Gesture Recognition for Sign Language Alphabets using Edge Oriented Histogram and Multi Class SVM](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=04de47bdd4a0753b33866a7cc445e6817e7a264d) papper as a reference for us."
date: \today
toc: true
---


\pagebreak
# 1. Pipeline

## 1.1. Input Module: (returns a list of image paths)
- Uses PIL module to read images from the disk and exclude them if corrupted.
- Sort the files in increasing order of the file name (as integers).
- Split the dataset into training, validation, and testing sets (70%, 10%, 20%).
   
## 1.2. Preprocessing Module:
- Read the image using opencv, then resize it to 200x200 pixels.
- Convert the image to grayscale.
- Segment the hand from the background using a skin detection (HSV), thresholding technique (Binary + OTSU).
- Morphological operations (Erosion + Dilation) to remove noise.
- Canny edge detection to detect the edges of the hand (remove useless information).

## 1.3. Feature Extraction Module:
- Used Edge of Oriented Histograms (EOH) to extract features from the image.

## 1.4. Classification Module:
- Used RandomizedSearchCV to find the best hyperparameters for the classifier.
- Used SVC with the best hyperparameters.
- Tested with different classifiers.
  - KNN
  - Random Forest
  - Logistic Regression
  - ADABoost
  - 2-layer NN

## 1.5. Performance Analysis Module:
- Used confusion matrix to analyze the performance of the classifier.
![Confusion Matrix](https://media.discordapp.net/attachments/1107081102687993907/1108467501714657330/image.png?width=652&height=651)

- Used classification report to analyze the performance of the classifier (best accuracy= 94.5%).

## 1.6. Future Enhancements:
- Preprocessing: Use better methods to eliminate shadow (ML or DL).
- Classification: Use a CNN to extract features and classify the images.



\pagebreak
# 2. Workload Distribution

## 2.1. Mai: 
- Image preprocessing

## 2.2. Rana: 
- Image preprocessing

## 2.3. Omar: 
- Input utils, Feature Extraction

## 2.4. Youssef:
- Classification, Output Utils
