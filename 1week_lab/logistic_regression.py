import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load dataset
data = load_diabetes()
X = data.data[:, :2]
Y = data.target

# Convert regression target to binary
Y = (Y > np.mean(Y)).astype(int)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=45
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias column
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]


class LogisticRegression:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, Y, Y_pred):
        ep = 1e-15
        Y_pred = np.clip(Y_pred, ep, 1-ep)
        return -np.mean(Y*np.log(Y_pred) +
                        (1-Y)*np.log(1-Y_pred))

    def fit(self, X, Y, X_test, Y_test):

        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)

        self.train_losses = []
        self.test_losses = []

        for _ in range(self.epochs):

            
            z = np.dot(X, self.theta)
            Y_pred = self.sigmoid(z)

            # Loss
            train_loss = self.loss(Y, Y_pred)
            self.train_losses.append(train_loss)

            test_pred = self.sigmoid(np.dot(X_test, self.theta))
            test_loss = self.loss(Y_test, test_pred)
            self.test_losses.append(test_loss)

            # Gradient
            dw = (1/n_samples) * np.dot(X.T, (Y_pred - Y))

            # Update
            self.theta -= self.lr * dw

    def predict(self, X):
        z = np.dot(X, self.theta)
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)


# Train
model = LogisticRegression(lr=0.1, epochs=1000)
model.fit(X_train, Y_train, X_test, Y_test)

# Predict
predictions = model.predict(X_test)
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

# Plot Loss
plt.plot(model.train_losses)
plt.plot(model.test_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Test Loss"])
plt.show()
