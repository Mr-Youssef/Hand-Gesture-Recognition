#!/usr/bin/env python
# coding: utf-8

# ### All Imports

# In[ ]:


import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pickle import dump, load


# ### Multi-class SVM Model

# In[ ]:


def hyperparamter_tuning_svm(train_features, train_labels, validation_features, validation_labels):
    
    parameters = {'kernel':['rbf', 'poly', 'sigmoid'], 'C':range(1, 10), 'gamma':[i * 0.1 + 0.1 for i in range(10)], 'decision_function_shape':['ovr', 'ovo']}
    svc = svm.SVC()
    clf = RandomizedSearchCV(svc, parameters, scoring='accuracy', verbose = 2, refit = True)
    # clf.scorer_ = sklearn.metrics.make_scorer(accuracy_score)
    clf.fit(train_features, train_labels)
    sorted(clf.cv_results_.keys())
    y_pred = clf.predict(validation_features)
    accuracy = accuracy_score(validation_labels, y_pred)
    return clf.best_params_, accuracy


# In[ ]:


def SVC_Model(train_features, train_labels, validation_features, validation_labels, params):
    # Train the SVM model
    svc_model = SVC(kernel=params['kernel'], gamma = params['gamma'], C=params['C'], decision_function_shape=params['decision_function_shape'])
    svc_model.fit(train_features, train_labels)

    # Predict the class labels on the test set
    y_pred = svc_model.predict(validation_features)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(validation_labels, y_pred)
    return accuracy, y_pred, svc_model


# ### Confusion Matrix

# In[ ]:


def display_confusion_matrix(labels, y_pred):
    classes = ['0', '1', '2', '3', '4', '5']
    cm = confusion_matrix(labels, y_pred, labels=classes)

    conf_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(conf_matrix, cmap='Blues')

    # set the axis labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # set the text labels in each cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, format(conf_matrix[i, j], '.2f'),
                        ha="center", va="center", color="white")

    # set the plot title and show the color bar
    ax.set_title("Confusion Matrix", fontsize=16)
    plt.colorbar(im)

    # display the plot
    plt.show()
    


# In[ ]:


def saving_weights_pickle(model, name):
    dump(model, open(f'Models/{name}.pkl', 'wb'))


# In[ ]:


def classification(train_features, train_labels, validation_features, validation_labels):
    best_params, accuracy = hyperparamter_tuning_svm(train_features, train_labels, validation_features, validation_labels)
    accuracy, y_pred, svc_model= SVC_Model(train_features, train_labels, validation_features, validation_labels, best_params)
    display_confusion_matrix(validation_labels, y_pred)
    saving_weights_pickle(svc_model, 'SVC_Model')
    return accuracy


# In[1]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script classification_utils.ipynb')


# In[2]:


if __name__ == '__main__':
    create_py()


# In[ ]:




