from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Reshape, Add
from keras.models import Model, Sequential
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist, cifar10
import numpy as np

K.set_learning_phase(1) #set learning phase
batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    channel_shift_range= 1.0)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(96, kernel_size=(3, 3), strides = 1,
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(96, (3, 3), strides = 1,activation='relu'))
model.add(Conv2D(96, (3, 3), strides = 2,activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(192, (3, 3), strides = 1,activation='relu'))
model.add(Conv2D(192, (3, 3), strides = 1,activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(192, (3, 3), strides = 2,activation='relu'))
model.add(Conv2D(192, (3, 3), strides = 2,activation='relu'))
model.add(Conv2D(192, (1, 1), strides = 1,activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(10, (1, 1), strides = 1,activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks = [keras.callbacks.ModelCheckpoint("cnn.hd5", monitor='val_loss',
          verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#RESNET ------------------------------------------------------------------------------

def res_loss_function(y_true, y_pred, alpha=0.9):
    y_true = Reshape((32,32,3))(y_true)
    print(y_true.shape)
    baseline = model(y_pred)
    adverse = model(y_true)
    classifier_crossentropy = keras.losses.categorical_crossentropy(baseline, adverse)

    generative_crossentropy = keras.losses.binary_crossentropy(y_true, y_pred)
    print(generative_crossentropy.shape)
    generative_crossentropy = K.expand_dims(generative_crossentropy, 3)
    print(generative_crossentropy.shape)

    euc_distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=1))
    print(euc_distance.shape)
    euc = K.expand_dims(euc_distance, 3)
    print(euc.shape)

    out = ((1 - alpha) * (1/classifier_crossentropy)) + (alpha * generative_crossentropy ** 2)

    return out




input_img = Input(shape=(32, 32, 3))

flattened= Flatten()(input_img)
dense1 = Dense(3072, activation='relu')(flattened)
noise = keras.layers.GaussianNoise(0.1)(dense1)
dense2 = Dense(3072, activation='relu')(noise)
res_layer1 = Add()([dense2, flattened])
dense3 = Dense(3072, activation='relu')(res_layer1)
noise = keras.layers.GaussianNoise(0.1)(dense3)
dense4 = Dense(3072, activation='relu')(noise)
noise = keras.layers.GaussianNoise(0.1)(dense4)
res_layer2 = Add()([noise, res_layer1])
decoded = Reshape((32,32,3))(res_layer2)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


#autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)


decoded_imgs = autoencoder.predict(x_test)

autoencoder.compile(optimizer= keras.optimizers.Adadelta(), loss=res_loss_function)

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=
                [keras.callbacks.ModelCheckpoint("autoencoder.hd5", monitor='val_loss',
                verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

#autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)

#subset = np.random.randint(10000, size=128)
model = load_model('cnn.h5')
autoencoder = load_model('autoencoder.h5')
predictions = model.predict(x_train)
decoded_imgs = autoencoder.predict(x_train)
predictions1 = model.predict(decoded_imgs)

n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    print("Test")
    print(predictions[i])
    print("Adv")
    print(predictions1[i])
    plt.imshow(x_train[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("test.png")
# plt.show()
