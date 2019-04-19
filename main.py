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
print(myData.head())                    # heh



myData.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
print(myData.head())

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
print(myData.head())
print(myData.dtypes)



myData = pd.get_dummies(myData, drop_first=True)
print(myData.dtypes)
print(myData.head())


# Categories have been spread out in a 'one-hot' encoding
# Age, blood pressure, cholesterol, etc are still numbers, not one-hot
# We're going to want to normalize these categories to either [0,1] or [-1,1]

myData = myData / myData.max()
# Quick/easy normalization.  Worth trying one which centers at 0... -OR- using a BN layer.

X_train, X_test, Y_train, Y_test = train_test_split(myData.drop('target', 1), myData['target'], test_size=.2, random_state=0)


model = Sequential()
model.add(Dense(100, input_shape=(19,)))
model.add(ReLU())
#model.add(Dropout(0.3))
model.add(Dense(100))
model.add(ReLU())
#model.add(Dropout(0.3))
model.add(Dense(10))
model.add(ReLU())
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='Adam', loss='categorical_hinge', metrics=['accuracy'])
### Categorical Hinge Loss is pretty good...
### Recall that Hinge loss is a direct generalization of SVM, i.e. we're not just looking to classify,
### but we want our data points to be as far from a boundary as possible.

#model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=150)

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
#print(myData.describe())
#print(myData.info())
#print(myData.shape)             # Rows then Columns

plt.figure(figsize=(10,10))
sns.heatmap(myData.corr(), annot=True, fmt='.1f')
plt.show()

