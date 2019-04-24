TODO:
 - Non NN models   (Done now, still need to describe what's going on in readme.md)
 - Analysis of best models
 - Domain knowledge
 - Some pretty graphs
 - PyTorch implementation
 - Intro and Outro sections   (Intro done now)
 - After everything else, update code on readme.md to reflect final python code


# Intro:

This real-world dataset was found on Kaggle, and contains data on 303 patients from (1) The Hungarian Institute of Cardiology, (2) University Hospital, Zurich, (3) University Hospital, Basel, (4) V.A. Medical Center, Long Beach, and (5) The Cleveland Clinic Foundation.  This dataset was donated to the greater scientific community in 1988 and has since been cited by dozens of academic papers and used as a sort of testing sandbox for new ideas in machine learning.  

The data consists of 14 attributes.  Some attributes are continuous, like age and cholesterol level.  Some are categorical, like sex and type of chest pain.  The task is to use the first 13 attributes to predict the 14th - the presence of heart disease in a patient.  This task is complicated by the fact that the training set is so small.  We have to be very careful because even a modest (by todays standards) neural network has more degrees of freedom than there are data points in our set!

In this project we'll explore various ML models for this task.  Our main tools will be the python scikit-learn and tensorflow libraries.  It should be stated straightaway, however, that we should not hope for a model with 100% accuracy - indeed, we should be skeptical of any model that does too well, as its success will likely be the result of a large parameter space. What we will find is quite reassuring:  most models achieve an accuracy of .885.  This consistency across many types of model suggests a Bayes Error probably around 8-10%.  



# Getting Started: Imports and Data Preparation
Let's go ahead and import our modules and load our training data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, ReLU, Dropout

pd.options.mode.chained_assignment = None            # Get rid of warning messages
myData = pd.read_csv("heart.csv")
```

Next (Optional): Rename data fields to be more descriptive.

```python
myData = pd.read_csv("heart.csv")
myData.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
myData['sex'][myData['sex'] == 0] = 'female'
myData['sex'][myData['sex'] == 1] = 'male'
myData['chest_pain_type'][myData['chest_pain_type'] == 1] = 'typical angina'
myData['chest_pain_type'][myData['chest_pain_type'] == 2] = 'atypical angina'
myData['chest_pain_type'][myData['chest_pain_type'] == 3] = 'non-anginal pain'
myData['chest_pain_type'][myData['chest_pain_type'] == 4] = 'asymptomatic'
myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
myData['fasting_blood_sugar'][myData['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'
myData['rest_ecg'][myData['rest_ecg'] == 0] = 'normal'
myData['rest_ecg'][myData['rest_ecg'] == 1] = 'ST-T wave abnormality'
myData['rest_ecg'][myData['rest_ecg'] == 2] = 'left ventricular hypertrophy'
myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 0] = 'no'
myData['exercise_induced_angina'][myData['exercise_induced_angina'] == 1] = 'yes'
myData['st_slope'][myData['st_slope'] == 1] = 'upsloping'
myData['st_slope'][myData['st_slope'] == 2] = 'flat'
myData['st_slope'][myData['st_slope'] == 3] = 'downsloping'
myData['thalassemia'][myData['thalassemia'] == 1] = 'normal'
myData['thalassemia'][myData['thalassemia'] == 2] = 'fixed defect'
myData['thalassemia'][myData['thalassemia'] == 3] = 'reversible defect'
```

Important: Use one-hot encoding for categorical variables.
```python
myData = pd.get_dummies(myData, drop_first=True)
```

Prepare the data:  Neural networks train much (much, much) quicker on normalized datasets.  
It's also important to reserve part of the dataset as a test set.  Unfortunately this dataset is very small, so we can't really afford to set aside a validation set.  Let's reserve 20% of our data as a test set.
```python
myData = (myData - np.min(myData))/(np.max(myData) - np.min(myData))
x = myData.drop('target', axis=1)
y = myData['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=2)
```



# Scikit-Learn Models
Scikit-learn enables us to quickly build and train models to a dataset.  Each of these models has a fit() method, as well as predict() and score().  This uniformity will allow us to make an ensemble voting classifier without much effort.

```python
lin_model = LogisticRegression(solver='lbfgs')
lin_model.fit(x_train, y_train)
print("Linear Model Accuracy: ", lin_model.score(x_test, y_test))

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
print("K Nearest Neighbor Model Accuracy: ", knn_model.score(x_test, y_test))

svm_model = SVC(gamma='auto')
svm_model.fit(x_train, y_train)
print("Support Vector Machine Model Accuracy: ", svm_model.score(x_test, y_test))

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
print("Naive Bayes Model Accuracy: ", nb_model.score(x_test, y_test))

tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
print("Decision Tree Model Accuracy: ", tree_model.score(x_test, y_test))

forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(x_train, y_train)
print("Random Forest Model Accuracy: ", forest_model.score(x_test, y_test))
```

Outputs will vary (quite a bit, actually!) depending on random seeds.  One run produced the following output:
```
Linear Model Accuracy:  0.8524590163934426
K Nearest Neighbor Model Accuracy:  0.8360655737704918
Support Vector Machine Model Accuracy:  0.8524590163934426
Naive Bayes Model Accuracy:  0.8688524590163934
Decision Tree Model Accuracy:  0.7540983606557377
Random Forest Model Accuracy:  0.8360655737704918
```


# Deep Learning Model

We'll build a simple model with three fully-connected layers of 100 units, 100 units, 10 units, and ReLU activations.  This last layer feeds into a single unit with sigmoid activation.  The best choices for loss function for a classification task are typically binary_crossentropy and categorical_hinge.  Categorical_hinge is a direct generalization of SVM, and happens to work well here, so that's what we'll go with.
```python
model = Sequential()
model.add(Dense(100, input_shape=(19,)))
model.add(ReLU())
model.add(Dense(100))
model.add(ReLU())
model.add(Dense(10))
model.add(ReLU())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='Adam', loss='categorical_hinge', metrics=['accuracy'])
```


Train the model:
```python
model.fit(X_train, Y_train, epochs=15)
```


And check how it does on our test set:
```python
y_predicted = (model.predict(X_test) >= 0.5)

conf_mat = confusion_matrix(Y_test, y_predicted)
print(conf_mat)
total = sum(sum(conf_mat))
sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total

print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
print('accuracy : ', accuracy)
```

Here's the console output:
```
[[23  4]
 [ 3 31]]
specificity :  0.8857142857142857
sensitivity :  0.8846153846153846
accuracy :  0.8852459016393442

```

Some analysis:  The confusion matrix is interpreted as follows:  Upon running our learned model on the test set, the model came up with 23 true negatives and 31 true positives.  There were 4 false positives and 3 false negatives, leading to specificity of 31/35 and sensitivity of 23/26.  It must be noted that since the test set is so small (61 samples), every right or wrong answer alters the accuracy score by about 1.5%.  This volatility is very undesirable and heavily rewards overfitting to the test set - again, something which could be avoided if we had enough data for a validation set.

Since this model predicts something as important as heart disease, we're much happier with false positives than with false negatives.  There are several ways to decrease the number of false negatives.  Recall that this neural network has a last-layer sigmoid, so the activation of the final neuron is somewhere between 0 and 1.  By default we round to the nearest integer to obtain a prediction, so that (for example) if some input to the network leads to a final neuron activation of 0.6, we predict heart disease, and if some input leads to a final activation of 0.4, we predict no heart disease.  Instead of rounding to the nearest integer (i.e. returning (final_activation >= 0.5) ), we could change the cutoff point to some other value.  Consider the following variant:

```python
y_predicted = (model.predict(x_test) > 0.15)
```

We now predict heart disease in individuals whose data leads to a final neuron activation of 0.15 or more.  We may have caught some of the patients who would have fallen through the cracks using the stricter cutoff of 0.5.  Indeed, after this change we obtain the following confusion matrix, specificity, sensitivity, accuracy:

```
[[17 10]
 [ 1 33]]
specificity :  0.7674418604651163
sensitivity :  0.9444444444444444
accuracy :  0.819672131147541
```

Many more false positives to be sure, but only one false negative.  

Another way to decrease false negatives is to create an ensemble voting model out of many simpler models, and to require not just a majority vote, but a supermajority vote, to declare a negative result.




After looking through some kaggle kernels on this problem, I noticed that many models achieved test accuracy of .885.  This number was consistent across:  Random forest, decision tree, k-nearest neighbor, SVM, logistic regression, deep and shallow NN, etc.  The fact that so many classifiers got to the same 88.5% accuracy suggests that the Bayes error for this task may well be close to 11.5%.  





# Closing Remarks:
While looking into this dataset I noticed that the 'target' variable may have been entered incorrectly.  The 'target' variable is supposed to be 0 if there's no heart disease and 1 if there is heart disease.  However, looking at the following heat map of the data set suggests that it might be the other way around:

 ![Heatmap](/Images/heatmap.png)
 
 We see that 'target' is negatively correlated with age, cholesterol level, being male, ... All things which one would think make heart disease more likely, not less.  I decided to hold this until the conclusion because it doesn't change *how* one would carry out this analysis; it simply changes the model one arrives at.  
 




# (Sketch) Closing remarks:  
 - Bayes error is probably somewhere around 8-10%
 - naive Bayes model achieves nearly 92% accuracy, but this may be due to random chance plus not having a validation set, and is almost certainly helped by the fact that the test set is tiny, so getting just one more correct prediction corresponds to a large percentage jump in accuracy. 
  - Random Forest with 100 estimators achieves 90% accuracy.  This accuracy falls with more estimators.  (Note to self, we should do a search to find the optimal number of estimators)
  - logistic regression, k-nearest-neighbor, SVM, and Neural Net all achieve 85.5% accuracy.  
  - Some words on the least and most important features when it comes to predicting heart disease.



