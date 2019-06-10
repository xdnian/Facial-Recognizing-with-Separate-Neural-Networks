import os
import sys
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

from dataset import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training parameters
batch_size = 30
epochs = 15
interval = 20

def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=(INPUT_SIZE, INPUT_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(NUM, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def load_all_dataset():
    x_train = []
    y_train = []    
    x_test = []
    y_test = []

    dirs = next(os.walk(DATASET_PATH))[1]
    
    for d in dirs:
        files = os.listdir(os.path.join(DATASET_PATH, d))
        shuffle(files)

        label = LABEL.index(d)

        for i in range(3):
            x_test.append(load_grey_image(d, files[i]))
            y_test.append(label)
        for i in range(3, len(files)):
            x_train.append(load_grey_image(d, files[i]))
            y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, NUM)
    y_test = np_utils.to_categorical(y_test, NUM)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test 

def train_model():

    x_train, y_train, x_test, y_test = load_all_dataset()
    model = create_model()

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
        #   validation_split=0.1
          validation_data=(x_test, y_test)
        )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def train_and_evaluate_model():
    Start = time.time()

    x_train, y_train, x_test, y_test = load_all_dataset()
    model = create_model()
    train_scores = []
    val_scores = []

    for i in range(interval):
        print(i, '/', interval)
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_data=(x_test, y_test))
        # hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.1)
        if i == 0:
            train_scores.append(hist.history['acc'][0])
            val_scores.append(hist.history['val_acc'][0])
        train_scores.append(hist.history['acc'][epochs-1])
        val_scores.append(hist.history['val_acc'][epochs-1])

    print("\nTEST")
    loss, matrics = model.evaluate(x_test, y_test, verbose=0)
    print("- loss:", loss, "- acc:", matrics)

    End = time.time()
    print(round(End-Start, 3))

    # Save the model
    model.save(time.strftime('./new/model/classification_model.h5'))

    plt.plot(range(interval+1), train_scores,
             color='blue', marker='o',
             markersize=5, label='training ACC')

    plt.plot(range(interval+1), val_scores,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation ACC')

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend(loc='lower right')
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    # Save the train process
    # plt.savefig(time.strftime('./new/figure/classification_model.png'), dpi=300)
    plt.show()


if __name__=='__main__':

    # train_model()
    train_and_evaluate_model()


    