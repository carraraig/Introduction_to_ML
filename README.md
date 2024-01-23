# Artificial intelligence : Introduction to machine learning

This course will develop an introduction to ML, by reviewing the fundamental principles and methods. Each lecture will be accompanied by a hands-on practical (in python), during which datasets of biological and/or medical importance will be processed. Doing so will provide a unique opportunity to assess the performances of the various methods studied (running time, stability, sensitivity to noise/outliers, etc), and to think critically about the quality of models in biology/medicine.

## Lesson 1 - Introduction to machine learning
This lecture will introduce the main ingredients of ML, namely the different classes of problems, the data involved in such processes, the main classes of algorithms, and the learning process.

Topics :
- Data types
- Supervised vs non-supervised learning
- Algorithms taxonomy
- Software platforms and languages

Practical/potential applications :
- Data manipulations
- Model complexity and under/over fitting
- The bias-variance trade-off
- Jupyter Notebook
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Lesson 2 - Linear regression 

Regression is the problem concerned with the prediction a response value from variables. This course will cover the basics of the method including the selection of variables and the design of sparse models.

Topics :
- Linear regression and least squares 
- Errors and model adequacy 
- Sparse models

Practical/potential application :
- Underfit and Overfit
- Dataset Split
- Diabetes Detection

## Lesson 3 - Classification with Logistic Regression

Logistic regression is a supervised classification algorithm used to model the probability of an observation to belong to a given class. To do so, a linear model is used to estimate the parameters.

Topics :
- Classification using linear models
- The logistic regression

Practical/potential application :
- Balance and Unbalance dataset
- Stratified Train Test Val Split
- Cross Validation
- Application on IRIS dataset
- Application on MNIST dataset

## Lesson 4 - Support Vector Machines

SVM are a popular and robust class of models to perform supervised classification. The main difficulties are to deal with classes which are partially mixed -- e.g. due to noise, and whose boundaries have a complex geometry.

Topics :
- Linear separability and support vectors
- Soft margin separators 
- Kernels and non linear separation 
- Multiclass classification

Practical/potential application :
- Soft and Hard Margin
- Non Linear Kernel
- Unbalanced Dataset
- Application on classification of cell types

## Lesson 5 - Bayesian statistics

LDA is another supervised classification algorithm using a linear combination of features defining boundaries separating two or more classes. This lecture will introduce LDA and compare it to the so-called Naive Bayes classifier.

Topics :
- Naive Bayes classifier
- LDA

Practical/potential application :
- Naive Bayes classifier
- LDA
- GridSearchCv

## Lesson 6 - Dimensionality Reduction

Dimensionality reduction methods aim at embedding high-dimensional data into a lower-dimensional space, while preserving specific properties such as pairwise distances, the data spread, etc. Originating with the celebrated Principal Components Analysis method, recent methods have focused on data located on non linear spaces.

Topics :
- Principal Component Analysis (PCA)
- t-Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP)

Practical/potential application :
- PCA
- Criterion PCA
- t-SNE
- Perplexity parameter
- UMAP
- Application on IRIS dataset
- Application on MNIST dataset

## Lesson 7 - Clustering

In a non supervised context, clustering aims at grouping the data in homogeneous groups by minimizing the intra-group variance.  This fundamental task is surprisingly challenging due to several difficulties: the (generally) unknown number of clusters, clusters whose boundaries have a complex geometry, dealing with overlapping clusters (due to noise), dealing with high dimensional data, etc. This class will present two main clustering techniques:

Topics :
- k-means and k-means++
- Hierarchical clustering

Practical/potential applications :
- Evaluation Metric
- k-means
- Spectral Clustering
- Hierarchical Clustering
- Plot Dendrogramm


