from tensorflow import keras

"""CNN stands for convolutional neural networks 
Then consist of convolutional layers 
they are better than dense layer powered neural networks at classification. 
They will be able to keep track of features anywhere on the image 
"""


class EndEpochCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy',0)>0.89:
            self.model.stop_training = True


fm = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fm.load_data()

ee_cb = EndEpochCallback()

train_images = train_images/255.0  #Normalizing Data

print("Train Images Shape",train_images.shape)
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images/255.0
test_images = test_images.reshape(-1,28,28,1)

# Building a simple CNN architecture

model = keras.Sequential([keras.layers.Conv2D(64, (3, 3), activation='relu',
                                              input_shape=(28, 28, 1)),
                          keras.layers.MaxPooling2D(2, 2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')
                          ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, callbacks=[ee_cb], batch_size=512)

model.evaluate(test_images, test_labels)

### Keep Batch Size at a minimum # it hits desired accuracy in the first run
