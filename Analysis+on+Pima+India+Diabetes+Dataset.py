
# coding: utf-8

# In[65]:

#Dependencies
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

#Loading Data
data = pd.read_csv("C:\\Users\\Sayak\Desktop\\Pima_Indians_Diabetes\\pima-indians-diabetes.csv")
# https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names

# Plotting the original classification
data.hist(column="1",       
              figsize=(8,8),         
              color="blue") 
plt.title('Initial Distribution')
plt.show()

#Assigning the Features except the class label
x = data.ix[:,:-1].values
#Standardizing them
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)
#Assigning the class label
y = data.ix[:,-1].values
# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
print ("The Accuracy Scores w.r.t the neighbors chosen:\n")
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_std, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[27]:

#PCA for a bit of dimensionality reduction and then Logistic Regression which is working a bit better
def doPCA():
    pca = PCA(n_components=7)
    pca.fit(x_std)
    return pca

pca = doPCA()
x_transformed = pca.transform(x_std)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, x_transformed, y, cv=10, scoring='accuracy').mean())


# In[43]:

#Support Vector Machines with a linear kernel 
from sklearn import svm
clf = svm.SVC(kernel='linear', shrinking=False)
print(cross_val_score(clf, x_transformed, y, cv=10, scoring='accuracy').mean())


# In[51]:

#SGD with Logistic Regression
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss='log',random_state=10)
print(cross_val_score(clf, x_transformed, y, cv=10, scoring='accuracy').mean())


# In[52]:

#SGD with Logistic Regression
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss = 'hinge', random_state=10)
print(cross_val_score(clf, x_transformed, y, cv=10, scoring='accuracy').mean())

