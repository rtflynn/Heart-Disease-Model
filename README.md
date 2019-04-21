TODO:
 - Non NN models
 - Analysis of best models
 - Domain knowledge
 - Some pretty graphs
 - PyTorch implementation
 - Intro and Outro sections
 - After everything else, update code on readme.md to reflect final python code


Getting started:  Import modules and load our training data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, ReLU, Dropout
np.random.seed(100)

pd.options.mode.chained_assignment = None

myData = pd.read_csv("heart.csv")
```

Next (Optional): Rename field values to be more descriptive.

```python
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

Important:  Use one-hot encoding for categorical variables.
```python
myData = pd.get_dummies(myData, drop_first=True)
```

Prepare the data:  Neural networks train much (much, much) quicker on normalized datasets.  
It's also important to reserve part of the dataset as a test set.  Unfortunately this dataset is very small, so we can't really afford to set aside a validation set.  Let's reserve 20% of our data as a test set.
```python
myData = myData / myData.max()
X_train, X_test, Y_train, Y_test = train_test_split(myData.drop('target', 1), myData['target'], test_size=.2, random_state=0)
```

Build the model.  This is a simple model with two fully-connected layers of 100 units each, and ReLU activations.  The best choices for loss function for a classification task are typically binary_crossentropy and categorical_hinge.  Categorical_hinge is a direct generalization of SVM, and happens to work well here, so that's what we'll go with.
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
model.fit(X_train, Y_train, epochs=150)
```


And check how it does on our test set:
```python
y_predicted = model.predict(X_test)

conf_mat = confusion_matrix(Y_test, np.round(y_predicted))
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


Some analysis:  The confusion matrix is interpreted as follows:  Upon running our learned model on the test set, the model came up with 23 true negatives and 31 true positives.  There were 3 false positives and 4 false negatives, leading to specificity of 31/35 and sensitivity of 23/26.

After looking through some kaggle kernels on this problem, I noticed that many models achieved test accuracy of .885.  This number was consistent across:  Random forest, decision tree, k-nearest neighbor, SVM, logistic regression, deep and shallow NN, etc.  The fact that so many classifiers got to the same 88.5% accuracy suggests that the Bayes error for this task may well be close to 11.5%.  


(Chen: Opinion on the followin plz)

By the way, I think the data set has a problem.  The 'target' variable is supposed to be 0 if no heart disease, and 1 if heart disease, but I'm relatively (?) certain they reversed these values.  Check out the heat map.  Forget 'target' for a minute and look at the age column.  This looks like we'd expect, for example there's a 0.3 in cholesterol meaning we have a positive correlation between age and cholesterol.  We have -0.4 for max heart rate which again makes sense.

 ![Heatmap](/Images/heatmap.png)

'target' is negatively correlated with age, cholesterol, being male....   But I don't know anything about number of blood vessels, levels of pain, etc.  What do you think?



