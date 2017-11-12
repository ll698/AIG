from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
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
epochs = 10

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
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#RESNET ------------------------------------------------------------------------------

def res_loss_function(y_true, y_pred, alpha=0.5):
    y_true = Reshape((32,32,3))(y_true)
    print(y_true.shape)
    baseline = model(y_pred)
    adverse = model(y_true)
    classifier_crossentropy = keras.losses.categorical_crossentropy(baseline, adverse) ** 2

    generative_crossentropy = keras.losses.binary_crossentropy(y_true, y_pred)

    generative_crossentropy = K.expand_dims(generative_crossentropy, 3)

    euc_distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=1))
    euc_distance = K.expand_dims(euc_distance, 1)
    euc = K.expand_dims(euc_distance, 3)

    out = ((1 - alpha) * (1/classifier_crossentropy)) + (alpha * euc ** 4)

    return out




input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
print(x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)

flattened = Flatten()(x)
print(flattened.shape)
dense1 = Dense(512, activation='relu')(flattened)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(512, activation='relu')(dense3)
res_layer1 = Add()([dense4, dense1])
print(res_layer1.shape)
reshaped = Reshape((8,8,8))(res_layer1)
print(reshaped.shape)

# dense1 = Dense(64, activation='relu')(res_layer1)
# dense2 = Dense(64, activation='relu')(dense1)
# res_layer = Add()([dense2, res_layer1])
# dense3 = Dense(128, activation='relu')(res_layer)
# dense4 = Dense(128, activation='relu')(dense3)
# res_layer1 = Add()([dense3, dense4])
# reshaped = Reshape((4,4,8))(res_layer1)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(8, (3, 3), activation='relu', padding='same')(reshaped)
print(x.shape)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
print(x.shape)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded.shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


#autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)


decoded_imgs = autoencoder.predict(x_test)

autoencoder.compile(optimizer='adadelta', loss=res_loss_function)

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

#autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 128, epochs=epochs, verbose=1)

#subset = np.random.randint(10000, size=128)
decoded_imgs = autoencoder.predict(x_train)
predictions = model.predict(x_train)
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
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
