# MNIST Classifier based on AutoEncoder and KNN

This project implements a simple auto-encoder net, followed with KMeans classifier.


* The project tested with different hiper-parameters as: data split percentage, batch size, epochs number, clusters number.

* The results were analyzed using silhouette_score (average the density and separation of the samples in clusters).

* The model tested on test data and received similar results to the training data (no overfitting).

* The model trained on a small data set (and for short training) hence showing underfitting. 



The following files are included:
1. MNIST_AutoEncoderBasedClassifier.py - my "script" code
2. MNIST_AutoEncoderBasedClassifier.ipynb - Jupyter notebook 
* Packages required: Numpy, Keras, Matplotlib, Sklearn.  
 
