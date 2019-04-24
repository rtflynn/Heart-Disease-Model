import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, ReLU
import tensorflow as tf                # To remove logging
from statistics import mode

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.mode.chained_assignment = None
np.random.seed(100)

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

myData = pd.get_dummies(myData, drop_first=True)
# Categories have been spread out in a 'one-hot' encoding
# Age, blood pressure, cholesterol, etc are still numbers, not one-hot
# We're going to want to normalize these categories to either [0,1] or [-1,1]
myData = (myData - np.min(myData)) / (np.max(myData) - np.min(myData))
x_train, x_test, y_train, y_test = train_test_split(myData.drop('target', axis=1),
                                                    myData['target'], test_size=.2, random_state=0)


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


nn_model = Sequential()
nn_model.add(Dense(100, input_shape=(19,)))
nn_model.add(ReLU())
nn_model.add(Dense(100))
nn_model.add(ReLU())
nn_model.add(Dense(10))
nn_model.add(ReLU())
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='Adam', loss='categorical_hinge', metrics=['accuracy'])
nn_model.fit(x_train, y_train, epochs=15, verbose=0)


y_predicted = (nn_model.predict(x_test) > 0.5)
conf_mat = confusion_matrix(y_test, y_predicted)
print(conf_mat)
total = sum(sum(conf_mat))
sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total

print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
print('accuracy : ', accuracy)


votes = lin_model.predict(x_test) + svm_model.predict(x_test) + nb_model.predict(x_test) \
        + forest_model.predict(x_test) + tree_model.predict(x_test) + knn_model.predict(x_test) \
        + ((nn_model.predict(x_test)>0.5).T)[0].astype(float)

print(votes >= 4)

#print(lin_model.predict(x_test) + svm_model.predict(x_test) + nb_model.predict(x_test)
#      + forest_model.predict(x_test) + tree_model.predict(x_test) + knn_model.predict(x_test)
#      + ((nn_model.predict(x_test)>0.5).T)[0].astype(float))

print(y_test.values == 1.0)

conf_mat = confusion_matrix((y_test.values == 1.0),  (votes >= 4))
total = sum(sum(conf_mat))
sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total

print("Statistics for voting classifier, where simple majority rules:\n")
print(conf_mat)
print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
print('accuracy : ', accuracy)


conf_mat = confusion_matrix((y_test.values == 1.0),  (votes >= 2))
total = sum(sum(conf_mat))
sensitivity = conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])
specificity = conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])
accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/total

print("Statistics for voting classifier, where it only takes 2 positive votes (out of 7 votes) to declare "
      "a positive result:\n")
print(conf_mat)
print('specificity : ', specificity)
print('sensitivity : ', sensitivity)
print('accuracy : ', accuracy)