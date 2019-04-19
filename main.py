import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, ReLU
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
print(myData.head())                    # Categories have been spread out in a 'one-hot' encoding
# Age, blood pressure, cholesterol, etc are still numbers, not one-hot
# We're going to want to normalize these categories to either [0,1] or [-1,1]

X_train, X_test, Y_train, Y_test = train_test_split(myData.drop('target', 1), myData['target'], test_size=.2, random_state=10)

model = RandomForestClassifier(max_depth=5)
model.fit(X_train, Y_train)

estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = Y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:,1]
y_pred_bin = model.predict(X_test)


conf_matrix = confusion_matrix(Y_test, y_pred_bin)

print(conf_matrix)
total = sum(sum(conf_matrix))
sensitivity = conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0])
specificity = conf_matrix[1,1]/(conf_matrix[1,1] + conf_matrix[0,1])

print('specificity : ', specificity)
print('sensitivity : ', sensitivity)





### Okay, that was for random forests, which aren't sensitive to scale.
### If we want to use neural networks, we'll want to normalize our data first.

myData = myData / myData.max()  # also valid:   (myData - myData.mean()) / (myData.max() - myData.min())
print(myData.head(10))          # So now our data is normalized, all between 0 and 1

X_train = X_train / X_train.max()
X_test = X_test / X_test.max()
##### WHOOPS!  Fix this tomorrow!  We should never normalize train and test sets separately!!!


print(X_train.shape)
input_shape = X_train.shape[1]
print(input_shape)


model = Sequential()
model.add(Dense(100, input_shape=(19,)))
model.add(ReLU())
model.add(Dense(100))
model.add(ReLU())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.train_on_batch(X_train, Y_train)
for i in range(200):
    model.fit(X_train, Y_train)
print('OK')

y_predicted = model.predict(X_test)
print(y_predicted)

print('derp')
print(Y_test)

conf_mat = confusion_matrix(Y_test, np.round(y_predicted))
print(conf_mat)


print(conf_mat)
total = sum(sum(conf_mat))
sensitivity = conf_mat[0,0]/(conf_mat[0,0] + conf_mat[1,0])
specificity = conf_mat[1,1]/(conf_mat[1,1] + conf_mat[0,1])



print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
