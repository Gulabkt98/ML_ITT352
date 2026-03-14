import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss

data = load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

TP=np.sum((test_pred==1) & (Y_test==1))
TN=np.sum((test_pred==-1) & (Y_test==-1))
FP=np.sum((test_pred==1) & (Y_test==-1))
FN=np.sum((test_pred==-1) & (Y_test==1))

print("TP :",TP)
print("TN :",TN)
print("FP :",FP)
print("FN :",FN)      

## accuracy of test_data
                 
accuracy = ((TP+TN)/(TP+TN+FN+FP) )*100
print("accuracy :" ,accuracy )    

recall=(TP/(TP+FN))*100
print("reacall :",recall)

precision=(TP/(TP+FP))*100
print("precision :",precision)

f1__score= 2* (recall*precision)/(recall+precision)

print("f1_score :", f1__score)

model = LogisticRegression(max_iter=1, warm_start=True)

train_losses = []
test_losses = []

for i in range(100):
    model.fit(X_train, Y_train)

    train_prob = model.predict_proba(X_train)
    test_prob = model.predict_proba(X_test)

    train_losses.append(log_loss(Y_train, train_prob))
    test_losses.append(log_loss(Y_test, test_prob))

plt.plot(train_losses)
plt.plot(test_losses)
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.legend(["Train Loss", "Test Loss"])
plt.show()