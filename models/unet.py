from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def get_unet(input_shape=(3, 256, 256), optimizer=Adam()):
    inputs = Input(input_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu', init='he_normal', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', init='he_normal', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', init='he_normal', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', init='he_normal', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = merge([up5, conv3], mode='concat', concat_axis=1)
    conv5 = Convolution2D(128, 3, 3, activation='relu', init='he_normal', border_mode='same')(up5)
    conv5 = Convolution2D(128, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = merge([up6, conv2], mode='concat', concat_axis=1)
    conv6 = Convolution2D(64, 3, 3, activation='relu', init='he_normal', border_mode='same')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = merge([up7, conv1], mode='concat', concat_axis=1)
    conv7 = Convolution2D(32, 3, 3, activation='relu', init='he_normal', border_mode='same')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', init='he_normal', border_mode='same')(conv7)
    
    conv8 = Convolution2D(8, 1, 1, activation='relu', init='he_normal')(conv7)
    flat = GlobalAveragePooling2D()(conv8)
    image_probs = Activation('softmax', name='img_label')(flat)
    
    conv8_1 = Convolution2D(1, 1, 1, activation='relu', init='he_normal')(conv7)
    dense_probs = Activation('sigmoid', name='dense')(conv8_1)
    
    model = Model(input=inputs, output=[image_probs, dense_probs])
    model.compile(optimizer=optimizer, loss={'img_label': 'categorical_crossentropy', 'dense': dice_coef_loss},
                  metrics=['accuracy', 'categorical_crossentropy', dice_coef])
    
    model.summary()

    return model