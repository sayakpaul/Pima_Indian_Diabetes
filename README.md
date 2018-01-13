# Pima_Indian_Diabetes

The dataset details are available here: 
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names

What I did is pretty much straight forward. I took some well-known classifiers such as Logistic Regression, Support Vector Machines, 
kNN etc and compared and analyzed their performances on this dataset. Before that I standardized the dataset using our very own 
StandardScaler(). As with the classifiers I applied SGD (Stochastic Gradient Descent) with Logistic Regression and SVM for 
optimized training. I also applied PCA for Dimensionality Reduction. 

Among all of them Logistic Regression coupled with PCA seemed to be the best one.
