# machine-learning

This repository contains typical use cases of machine learning algorithms. Here I will show my work, and also generalize the mathematics so that their ability to solve problems are interpretable by your organization's needs.

### Linear Regression
The linear regression is a machine learning algorithm that takes linear data as input, and generates a line of best fit.
This algorithm is useful for business analytics such as forecasting, projections, and simple data analysis.

[Honey Production](https://github.com/nwoodr94/machine-learning/blob/master/Linear-Regression-scikit-learn.ipynb) is a project I coded using the linear regression algorithm, which took as input current production trends in beekeeping, and predicted future honey output.

### Multiple Linear Regression
The multiple linear regression is a machine learning algorithm that computes a line of best fit by regressing a function of multiple variables. Example use cases of this algorithm involve maximizing real estate returns given a rental properties many features, or predicting manufacturing output provided many variables.

[Rental Prices](https://github.com/nwoodr94/machine-learning/blob/master/Multiple-Linear-Regression-scikit-learn.ipynb) is a model I developed which uses a multiple linear regression algorithm to predict rental prices for apartments in New York City with 82% accuracy.

### Logistic Regression
The logistic regression is a machine learning algorithm which computes binary probabilities given specified features. For example, this algorithm could predict which students pass an exam, or what survival odds a person has in a natural disaster.

[Natural Disaster](https://github.com/nwoodr94/machine-learning/blob/master/Logistic-Regression-scikit-learn.ipynb) is a project I coded using a logistic regression algorithm, which isolated important features related to a natural disaster to predict that a human of my demographic had only a 10% chance of surviving the Titanic.

### Naive Bayes
The naive bayes is a machine learning and natural language processing algorithm which classifies data, frequently in the form of text. This algorithm uses probabilities to analyze sentiment in customer reviews of a product, or expectations of a particular user base.

[Text Classifier](https://github.com/nwoodr94/machine-learning/blob/master/Naive-Bayes-scikit-learn.ipynb) is a naive bayes classifier that determines the subject of an email using its word content with 99% accuracy. 

### K-Nearest Neighbors
The k-nearest neighbors is a machine learning algorithm which takes an unknown datapoint, and classifies it by polling the labels of its neighboring data. It is among the most powerful machine learning algorithms for future industries, deployed in recommendation engines and genetic sequencing technologies.

[MRI Scans](https://github.com/nwoodr94/machine-learning/blob/master/K-Nearest-Neighbors-scikit-learn.ipynb) is a project I coded using a k-nearest neighbors algorithm, which interpreted MRI scans to isolate and classify cancerous tumours with 96% accuracy.

### Decision Tree
The decision tree is a machine learning algorithm which iterates through the features of a new datapoint in order to classify it. An example use case of this algorithm is classifying complex objects, such as in a sorting or quality assurance application.

[National Flag](https://github.com/nwoodr94/machine-learning/blob/master/Decision-Tree-scikit-learn.ipynb) is an implemented decision tree algorithm, which interpreted feature data of a nation's flag, and with 55% accuracy predicted which continent that nation is located.

### Random Forest
The random forest is a ensemble learning algorithm that generates multiple decision trees, and polls the majority classification. It is considered a more robust approach to classification as it incorporates randomization techniques, and also computes feature importance. 

[Income Predictor](https://github.com/nwoodr94/machine-learning/blob/master/Random-Forest-scikit-learn.ipynb) is a project I coded using a random forest algorithm to predict which Americans make over $50,000 year with 82% accuracy.

### Perceptron
The perceptron is a machine learning algorithm that classifies linearly seperable data. It is the foundation of a neural network, and an ensemble of these algorithms constitute the building blocks of deep learning models.

[AND Gate](https://github.com/nwoodr94/machine-learning/blob/master/Perceptron-scikit-learn.ipynb) is a perceptron I developed which isolates the decision boundary inside a truth table to function as an AND logic gate.

### KMeans Clustering
The kmeans clustering is a machine learning algorithm which performs the k-nearest neighbors classification without human in-the-loop supervision. Typically for a desired outcome labels will be provided. This algorithm is capable of use cases including computer vision, or an otherwise unstructured k-nearest neighbors application.

1. [Digits](https://github.com/nwoodr94/machine-learning/blob/master/KMeans-Clustering-scikit-learn.ipynb) is a project I coded using a kmeans clustering algorithm, which interpreted my hand written numerical input and named the digits with varying accuracy.
2. [Iris](https://github.com/nwoodr94/machine-learning/blob/master/KMeans-Clustering-Iris-scikit-learn.ipynb) is a kmeans model that classifies subspecies of flora using only their dimensions with 92% accuracy.

# Deep Learning

Here are more advanced models trained using deep learning technologies like TensorFlow and Keras.

[Digits](https://github.com/nwoodr94/machine-learning/blob/master/deep-learning/deep_learning_MNIST.ipynb) is a deep learning model I developed which classifies numerical integers 0 - 10 with 97% accuracy.
