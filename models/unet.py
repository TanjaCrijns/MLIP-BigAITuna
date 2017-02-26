from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy, categorical_crossentropy

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true[:, 1, :, :])
    y_pred_f = K.flatten(y_pred[:, 1, :, :])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bin_cross(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.sum(binary_crossentropy(y_true, y_pred))

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
    
    conv8_1 = Convolution2D(2, 1, 1, activation='relu', init='he_normal')(conv7)
    conv8_1 = Lambda(lambda x: K.permute_dimensions(x))(conv8_1)
    dense_probs = Activation('softmax', name='dense')(conv8_1)
    
    model = Model(input=inputs, output=[image_probs, dense_probs])
    model.compile(optimizer=optimizer, 
                  loss={'img_label': 'categorical_crossentropy', 'dense': dice_coef_loss},
                  metrics={'img_label': ['accuracy', 'categorical_crossentropy'], 'dense': dice_coef},
                  loss_weights={'img_label': 0, 'dense': 1})
    
    model.summary()

    return model

def get_unet2(input_shape=(3, 256, 256), optimizer=Adam(lr=1e-5)):
    inputs = Input(input_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(2, 1, 1)(conv9)
    conv10 = Permute((2, 3, 1))(conv10)
    conv10 = Reshape((-1, 2))(conv10)
    conv10 = Activation('softmax')(conv10)
    conv10 = Reshape(input_shape[1:] + (2,))(conv10)
    conv10 = Permute((3, 1, 2))(conv10)

    model = Model(input=inputs, output=conv10)
    model.summary()

    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    return model