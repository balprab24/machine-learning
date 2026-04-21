# Machine Learning:

A collection of machine learning assingments/projects 

### Assignment 1 - Decision Trees
- Used scikit-learn to train and evaluate decision trees for binary classification
- Datasets: QSAR Biodegradation and PC4 NASA Software Defects
- Techniques: 10-fold cross-validation, pre-pruning, post-pruning
- Evaluated using AUC and ROC curves

### Assignment 2 - Comparing ML Models
- Used scikit-learn to compare multiple ML models for classification and regression
- Datasets: Contraceptive Method Choice (CMC) and Munich Rent Index 1999
- Techniques: 10-fold cross-validation, GridSearchCV hyperparameter tuning, MinMaxScaler normalization, one-hot encoding
- Classification models: Decision Tree, KNN, Multinomial Naive Bayes, Dummy Classifier
- Regression models: Decision Tree, KNN, Linear Regression, Dummy Regressor
- Evaluated using accuracy (classification) and RMSE/R2 (regression)

### Assignment 3 - Neural Networks (Keras)
- Implemented regression and classification using deep neural networks (Keras Sequential API)
- Datasets: Contraceptive Method Choice (CMC) and Munich Rent Index 1999
- Techniques: One-hot encoding, MinMaxScaler normalization, early stopping
- Regression: Compared architectures (2 vs 15 hidden layers, sigmoid vs relu)
- Classification: Compared architectures (2 vs 15 hidden layers, sigmoid vs relu)
- Evaluated using MSE (regression) and accuracy (classification)
- Plotted training/validation curves for all models