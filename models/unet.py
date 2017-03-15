from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy, categorical_crossentropy

smooth = 1.

def dice_coef(y_true, y_pred):
    """
    Compute the dice coefficient between two segmentations
    
    dice = (2*tp) / ((tp + fp) + (tp + fn))
    Only the 1st channel is used (binary segmentation)

    # Params
    - y_true : the ground truth prediction of shape 
               (batch_size, 2, width, height)
    - y_pred : predicted segmentation of shape
               (batch_size, 2, width, height)
    """
    y_true_f = K.flatten(y_true[:, 1, :, :])
    y_pred_f = K.flatten(y_pred[:, 1, :, :])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Compute the dice coefficient loss between two segmentations
    
    dice_loss = 1 - dice

    # Params
    - y_true : the ground truth prediction of shape 
               (batch_size, 2, width, height)
    - y_pred : predicted segmentation of shape
               (batch_size, 2, width, height)
    """
    return 1 - dice_coef(y_true, y_pred)

def bin_cross(y_true, y_pred):
    y_true_f = K.flatten(y_true[:, 1, :, :])
    y_pred_f = K.flatten(y_pred[:, 1, :, :])
    return K.sum(binary_crossentropy(y_true_f, y_pred_f))

def conv_bn_relu(inputs, n_filters, init):
    """
    Standard unet block: convolution, relu, batch_norm

    # Params
    - inputs : 4D input tensor
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor
    """
    conv = Convolution2D(n_filters, 3, 3, border_mode='same', init=init)(inputs)
    conv = Activation('relu')(conv)
    conv = BatchNormalization(axis=1, mode=2)(conv)
    return conv

def add_unet_block_cont(inputs, n_filters, init):
    """
    A Unet block in the first (contraction) stage
    conv, conv, pool

    # Params
    - inputs : 4D input tensor
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor after convs (for bridge to expansion step)
    - 4D output tensor after maxpool
    """
    conv1 = conv_bn_relu(inputs, n_filters, init)
    conv2 = conv_bn_relu(inputs, n_filters, init)
    pool = MaxPooling2D()(conv2)
    return conv2, pool

def add_unet_block_exp(inputs, bridge, n_filters, init):
    """
    A Unet block in the second (expansion) stage
    merge, upsample, conv, conv

    # Params
    - inputs : 4D input tensor
    - bridge: 4D tensor to concatenate
    - n_filters: number of filters in the conv layer
    - init : initialization

    # Returns
    - 4D output tensor after convs
    """    
    up = merge([UpSampling2D(size=(2, 2))(inputs), bridge], mode='concat', concat_axis=1)

    conv1 = conv_bn_relu(up, n_filters, init)
    conv2 = conv_bn_relu(conv1, n_filters, init)

    return conv2

def weighted_loss(losses):
    """
    Creates a single weighted loss function

    # Params
    - losses : a dictionary of (loss_function -> float),
    
    # Returns
    - A function taking y_true and y_pred that returns a weighted
      sum of the given losses.
    """
    def loss_func(y_true, y_pred):
        loss = 0
        for func, w in losses.items():
            loss += w * func(y_true, y_pred)
        return loss

    return loss_func

def get_unet(input_shape=(3, 256, 256), optimizer=Adam(lr=1e-5), 
              init='glorot_uniform', task='both',
              loss_weights=None):
    """
    U-net with batchnorm and (possibly) 2 heads
    https://arxiv.org/abs/1505.04597

    # Params
    - input_shape : input to the network
    - optimizer : optimizer to use
    - init : Initialization for conv layers
             Paper suggests he_normal initialization, but this did
             not work well in practice
    - task: either 'segm', 'label' or 'both'
    - loss_weights : weighting for losses
    """
    losses = {'label': 'categorical_crossentropy', 'segm': weighted_loss({dice_coef_loss: 4., bin_cross: 1})}
    metrics = {'label': ['accuracy', 'categorical_crossentropy'],
               'segm': ['accuracy', dice_coef, bin_cross]}

    inputs = Input(input_shape)

    bn1 = BatchNormalization(axis=1)(inputs)

    # Contraction
    conv1, pool = add_unet_block_cont(bn1, 32, init)
    conv2, pool = add_unet_block_cont(pool, 64, init)
    conv3, pool = add_unet_block_cont(pool, 128, init)
    conv4, pool = add_unet_block_cont(pool, 256, init)

    conv5 = conv_bn_relu(pool, 512, init)
    conv5 = conv_bn_relu(pool, 512, init)

    # Expansion
    conv6 = add_unet_block_exp(conv5, conv4, 256, init)
    conv7 = add_unet_block_exp(conv6, conv3, 128, init)
    conv8 = add_unet_block_exp(conv7, conv2, 64, init)
    conv9 = add_unet_block_exp(conv8, conv1, 32, init)

    outputs = []

    # Segmentation head
    if task != 'segm':
        label = Convolution2D(8, 1, 1, init=init)(conv9)
        label = Activation('relu')(label)
        label = GlobalAveragePooling2D()(label)
        label = Activation('softmax', name='label')(label)
        outputs.append(label)

    # Whole image label head
    if task != 'label':
        conv10 = Convolution2D(2, 1, 1, init=init)(conv9)
        if None not in input_shape:
            # This doesn't work when input_shape isn't fully defined
            # Either do it manually in numpy or fully
            # specify the input shape
            conv10 = Permute((2, 3, 1))(conv10)
            conv10 = Reshape((-1, 2))(conv10)
            conv10 = Activation('softmax')(conv10)
            conv10 = Reshape(input_shape[1:] + (2,))(conv10)
            conv10 = Permute((3, 1, 2), name='segm')(conv10)
            outputs.append(conv10)
    
    model = Model(input=[inputs], output=outputs)

    model.compile(optimizer=optimizer,
                  loss=losses[task] if task != 'both' else losses,
                  metrics=metrics[task] if task != 'both' else metrics,
                  loss_weights=loss_weights)

    model.summary()

    return model
