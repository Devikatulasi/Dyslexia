import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

data = pd.read_csv('https://raw.githubusercontent.com/Manvi-Kaur/HackathonML/main/labeled_dysx.csv')

data.head()

data.isnull().sum()

values = data.values
print(values)

data.boxplot(figsize=(7,5))

target = data['Label']

df = data.drop('Label',axis=1)

data

df

sorted(df)
xsmall = df.min()
q1, q3 = np.percentile(df, [25,75])
xlarge = df.max()

iqr = q3-q1
lb = q1-(1.5*iqr)
ub = q3+(1.5*iqr)

#df1 = df[df < ub]
#df1

#sns.boxplot(df1)

df1 = df[df > lb]
df1

sns.boxplot(df1)

#df1 = df[(df > lb) & (df < ub)]

sns.boxplot(df1)

from sklearn.model_selection import train_test_split

X = data.iloc[:, 0:6].values
Y = data.Label

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)

from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression()
LR_classifier.fit(X_train, y_train)

Predictions = LR_classifier.predict(X_test)

print(Predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Predictions)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,Predictions))

from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(cv=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(accuracy)

cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import mean_absolute_error
model_error = mean_absolute_error(y_pred, y_test)
model_error

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score

# Assuming 'y_true' are the true labels and 'y_pred' are the predicted labels
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
#average_precision_macro = average_precision_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1-Score:", f1_macro)
#print("Macro Average Precision:", average_precision_macro)



name = input("Enter name of applicant: ")
print("\nThe scores of all the tests in quiz as well as survey need to be entered.")
print("All the values lie in the range 0 to 1.\n")
lang_vocab = float(input("Enter the score of Language Vocab test: "))
memory = float(input("Enter the score of Memory test: "))
speed = float(input("Enter the score of Speed test: "))
visual = float(input("Enter the score of Visual Discrimination test: "))
audio = float(input("Enter the score of Audio Discrimination test: "))
survey = float(input("Enter the score obtained from Survey: "))

def get_result(lang_vocab, memory, speed, visual, audio, survey):
    array = np.array([[lang_vocab, memory, speed, visual, audio, survey]])
    label = int(model.predict(array))
    if label == 0:
        output = "There is a high chance of the applicant to have dyslexia."
    elif label == 1:
        output = "There is a moderate chance of the applicant to have dyslexia."
    else:
        output = "There is a low chance of the applicant to have dyslexia."
    return output

result = get_result(lang_vocab, memory, speed, visual, audio, survey)
print(result)

import pickle
pickle.dump(model, open('model1.pkl','wb'))

