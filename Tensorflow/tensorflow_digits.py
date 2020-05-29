from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

data = load_digits()

X = data.data
y = data.target

X = np.array([np.reshape(img, (8, 8)) for img in X])

NUM_TO_SHOW = 20

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

print([func for func in dir(keras.activations) if func[0] != '_'])

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(8, 8)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

loss, acc = model.evaluate(x_test, y_test)

print(acc, loss)

prediction = model.predict(x_test)

random_num = random.randint(NUM_TO_SHOW, len(prediction))

for p, real, data in zip(prediction[random_num-NUM_TO_SHOW: random_num], y_test[random_num-NUM_TO_SHOW: random_num], x_test[random_num-NUM_TO_SHOW: random_num]):
    plt.imshow(data, cmap='Greys')
    print('Answer: ', real)
    print('Prediction: ', np.argmax(p), '\n')
    plt.show()
