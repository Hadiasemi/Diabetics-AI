import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('diabetes.csv')
properties = list(df.columns.values)
properties.remove('Outcome')
X = df[properties]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

opt_name = ["nadam", "adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam"]
model_acc = []
for name in opt_name:
    model.compile(optimizer=name,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=500, batch_size=2)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    #print('Test accuracy:', test_acc)
    model_acc.append(test_acc)

max_acc = 0
max_index = 0
for i in range(len(model_acc)):
    if model_acc[i] > max_acc:
        max_index = i

model.compile(optimizer=opt_name[max_index],
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Highest acc. opt. name is:", opt_name[max_index])
print('Test accuracy:', test_acc)

                #return close to 0          #return close to 1
a = np.array([[5,117,86,30,105,39.1,0.251,42],[3,173,78,39,185,33.8,0.97,31]])
print(model.predict(a))

#close to zero means not diabetic
#close to one means is diabetic
