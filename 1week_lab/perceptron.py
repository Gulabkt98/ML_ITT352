import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=load_breast_cancer()

X=data.data[:,:2]
Y=data.target



Y=np.where(Y==0 ,-1 , 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

class perceptron :

    def __init__(self,lr=.01, epoch=1000):
            self.lr=lr
            self.epoch=epoch
    def fit(self, X , Y):
        n_examples, n_feature =X.shape
        self.theta=np.zeros(n_feature)
        self.theta_not=0
        
        for _ in range(self.epoch): 
            for i in range(n_examples):
                t=np.dot(self.theta,X[i])+self.theta_not
                y_pre=1 if t>=0 else -1

                if(y_pre!=Y[i]):
                     self.theta=self.theta+self.lr*Y[i]*X[i]
                     self.theta_not=self.theta_not+self.lr*Y[i]

    def predict(self ,X):
         y_pre=np.dot(X,self.theta)+self.theta_not
         return np.where(y_pre >=0 ,1,-1)

    def loss(self ,X,Y):
        y_pre=np.dot(X,self.theta)+self.theta_not
        loss=np.maximum(0,-Y*y_pre)
        return np.mean(loss)







model=perceptron()
model.fit(X_train,Y_train)        

predictions=model.predict(X_test)
print(predictions)


# print(X[:10])

print(model.loss(X_test,Y_test))




# import pandas as pd

# df=pd.DataFrame(data.data,columns=data.feature_names)
# df["target"]=Y
# print(df)


## confusion matrix for perceptron model 

TP=np.sum((predictions==1) & (Y_test==1))
TN=np.sum((predictions==-1) & (Y_test==-1))
FP=np.sum((predictions==1) & (Y_test==-1))
FN=np.sum((predictions==-1) & (Y_test==1))

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

## plot

## plot data points
plt.scatter(X_train[:,0] , X_train[:,1],c=Y_train)


## diceion boundary line
theta1 = model.theta[0]
theta2 = model.theta[1]
b = model.theta_not

x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

y_values = -(theta1/theta2) * x_values - b/theta2

plt.plot(x_values, y_values)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")

plt.show()




      

            



             
         
             
                
                    




