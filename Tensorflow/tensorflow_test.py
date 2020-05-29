import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# the images are the numerical data to give out network
# the labels are what they represent, print by indexing the list below e.g. class_names[train_labels[0]]
class_names = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# images are 2D arrays, where inner arrays are 1 row. each number is a greyscale value out of 255. We need to scale down to 1
train_images = train_images/255
test_images = test_images/255

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)), # input
        keras.layers.Dense(128, activation='relu'), # hidden layer
        keras.layers.Dense(10, activation='softmax') # output
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=4)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Accuracy: ', test_acc)

to_predict = test_images

prediction = model.predict(to_predict)

for count, p in enumerate(prediction):
    print('Answer: ', class_names[test_labels[count]])
    print('Prediction: ', class_names[np.argmax(p)], '\n')
