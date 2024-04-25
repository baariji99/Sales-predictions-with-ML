import keras.src.datasets.mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#loading data
(x_train,y_train),(x_test,y_test)=keras.src.datasets.mnist.load_data()
#scalling the data
x_train=x_train/255
x_test=x_test/255
print(x_train.shape,y_train.shape)
print(len(x_train),'x train lenth')
#printing the input
plt.matshow(x_train[0])
plt.show()
print(y_train[0:5])
#changing the shape of the data
x_train_flatten=x_train.reshape(len(x_train),28*28)
x_test_flatten=x_test.reshape(len(x_test),28*28)
print('x_train flatten',x_train_flatten.shape)
#creating neuran network model with no hiddened layer
model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy'])
model.fit(x_train_flatten,y_train,epochs=5)

print(y_test.shape,'y_test')
print(x_train_flatten.shape,'x_train_flatten')
model.evaluate(x_test_flatten,y_test)
plt.matshow(x_test[0])
plt.show()

y_predict=model.predict(x_test_flatten)
print(y_predict[0],'new predicted y')
print(np.argmax(y_predict[0]))
y_predicted_lebel=[np.argmax(i) for i in y_predict ]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_lebel)
#creating neuron network with hidden layer
print(cm)
model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')

])
model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy'])

model.fit(x_train_flatten,y_train,epochs=5)
model.evaluate(x_test_flatten,y_test)
y_predict1=model.predict(x_test_flatten)
print('y+predict+level1',y_predict1)
y_predicted_lebel1=[np.argmax(i) for i in y_predict1 ]
#creating confusion matrix
cm1=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_lebel1)
print(cm1)
sns.heatmap(data=cm,annot=True,fmt='d')
x_level='newx'
y_level='newy'
plt.show()
