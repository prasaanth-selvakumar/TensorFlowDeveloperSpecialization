from tensorflow import keras

fm = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fm.load_data()


class EpochEndCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):  # method gets called at the end of every epoch -
        # check tf docs for more methods like this
        if logs.get('accuracy', 0) > 0.85:
            self.model.stop_training = True


train_images = train_images/255.0  # Normalizing Data
test_images = test_images/255.0

eeCallback = EpochEndCallback()

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, callbacks=[eeCallback], batch_size=256)

model.evaluate(test_images, test_labels)
"""
The model evaluation will stop as soon as the target accuracy is achieved or 
it will continue until the number of epochs are over
This can be used to get models that satisfy our criteria

On the 10th epoch it hits an accuracy of 85 percent on the training data   
"""