# Importing Libraries
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt
import os
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Reshape
y_train=y_train.reshape(-1,)
y_classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# Observe the data
def show_image(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(y_classes[y[index]])
    plt.show()
x_train=x_train/255
x_test=x_test/255
# Model Building
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(4,4),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=34,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Train the model
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=10)
# Make Predictions
y_pred=model.predict(x_test)
y_pred[9]
y_pred=[np.argmax(i) for i in y_pred]
y_pred
y_test=y_test.reshape(-1,)
y_pred
show_image(x_test,y_test,9)
# Evaluate the model
model.evaluate(x_test,y_test)
from sklearn.metrics import classification_matrix
print(classification_matrix(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
import seaborn as sn
plt.figure(figsize=(14,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("Confusion Matrix")
plt.show()