import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models



initializer = {'class_name': 'VarianceScaling',
               'config': {'scale': 2.0, 'mode': 'fan_out', 'distribution': 'normal'}}

@tf.function
def activation_fn(x):
    return tf.nn.swish(x)


def separable_bn(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation(tf.nn.relu)(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), (stride, stride), dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False, kernel_initializer=initializer)(x)
    x = layers.BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(tf.nn.relu)(x)
    x = layers.Conv2D(filters, (1, 1), 1, 'same', use_bias=False, kernel_initializer=initializer)(x)
    x = layers.BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(tf.nn.relu)(x)

    return x


def conv(inputs, filters, k=1, s=1):
    x = layers.Conv2D(filters, k, s, 'same', use_bias=False, kernel_initializer=initializer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_fn)(x)
    return x


def se(inputs, filters, r):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(filters // (4 * r), 1, 1, 'same', activation=activation_fn, kernel_initializer=initializer)(x)
    x = layers.Conv2D(filters, 1, 1, 'same', activation='sigmoid', kernel_initializer=initializer)(x)

    return layers.multiply([inputs, x])


def residual(inputs, filters_in, filters_out, s, r, fused=True):
    if fused:
        x = inputs
        x = conv(x, filters_in * r, 3, s)
        x = conv(x, filters_out)
    else:
        x = conv(inputs, filters_in * r)
        x = layers.DepthwiseConv2D(3, s, 'same', use_bias=False, depthwise_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_fn)(x)
        x = se(x, filters_in * r, r)
        x = layers.Conv2D(filters_out, 1, 1, 'same', use_bias=False, kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
    if s == 1 and filters_in == filters_out:
        x = layers.add([x, inputs])

    return x


def backbone(inputs):
    feature = []
    filters = [24, 48, 64, 128, 160, 272, 1792]
    x = conv(inputs, filters[0], 3, 2)
    x = residual(x, filters[0], filters[0], 1, 1)
    x = residual(x, filters[0], filters[0], 1, 1)

    feature.append(x)

    x = residual(x, filters[0], filters[1], 2, 4)
    x = residual(x, filters[1], filters[1], 1, 4)
    x = residual(x, filters[1], filters[1], 1, 4)
    x = residual(x, filters[1], filters[1], 1, 4)

    feature.append(x)

    x = residual(x, filters[1], filters[2], 2, 4)
    x = residual(x, filters[2], filters[2], 1, 4)
    x = residual(x, filters[2], filters[2], 1, 4)
    x = residual(x, filters[2], filters[2], 1, 4)

    feature.append(x)

    x = residual(x, filters[2], filters[3], 2, 4, False)
    x = residual(x, filters[3], filters[3], 1, 4, False)
    x = residual(x, filters[3], filters[3], 1, 4, False)
    x = residual(x, filters[3], filters[3], 1, 4, False)
    x = residual(x, filters[3], filters[3], 1, 4, False)
    x = residual(x, filters[3], filters[3], 1, 4, False)
    x = residual(x, filters[3], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)
    x = residual(x, filters[4], filters[4], 1, 6, False)

    feature.append(x)

    x = residual(x, filters[4], filters[5], 2, 6, False)
    for _ in range(14):
        x = residual(x, filters[5], filters[5], 1, 6, False)

    feature.append(x)

    x = conv(x, filters[6])
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return feature, x


def spp_fn(x, rates=(6, 12, 18)):
    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization(epsilon=1e-5)(b0)
    b0 = layers.Activation(tf.nn.relu)(b0)

    b1 = separable_bn(x, 256, rate=rates[0], depth_activation=True, epsilon=1e-5)
    b2 = separable_bn(x, 256, rate=rates[1], depth_activation=True, epsilon=1e-5)
    b3 = separable_bn(x, 256, rate=rates[2], depth_activation=True, epsilon=1e-5)

    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Lambda(lambda y: backend.expand_dims(y, 1))(b4)
    b4 = layers.Lambda(lambda y: backend.expand_dims(y, 1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
    b4 = layers.BatchNormalization(epsilon=1e-5)(b4)
    b4 = layers.Activation(tf.nn.relu)(b4)
    b4 = layers.UpSampling2D(backend.int_shape(x)[1:3], interpolation='bilinear')(b4)

    return layers.concatenate([b4, b0, b1, b2, b3])


def build_model(shape=(256, 256, 3), classes=4):
    inputs = layers.Input(shape=shape)
    print(shape)
    features, _ = backbone(inputs)

    x = spp_fn(features[-2])

    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-5)(x)
    x = layers.Activation(activation_fn)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)

    x = layers.concatenate([x, conv(features[1], 48)])

    x = separable_bn(x, 256, depth_activation=True, epsilon=1e-5)
    x = separable_bn(x, 256, depth_activation=True, epsilon=1e-5)

    x = layers.Conv2D(classes, (1, 1), padding='same')(x)

    x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)

    return models.Model(inputs, x)


def segmentation_loss(class_weights):
    class_weights = tf.cast(tf.constant(class_weights), tf.float32)
    
    def weighted_cross_entropy(y_true, y_pred):
        import keras.backend as KB
        
#         y_true = KB.one_hot(tf.cast(KB.flatten(y_true), tf.int32),
#                             KB.int_shape(y_pred)[-1]+1)
#         unpacked = tf.unstack(y_true, axis=-1)
#         print(unpacked)
#         y_true = tf.stack(unpacked[:-1], axis=-1)
#         y_true = tf.reshape(y_true, [12,256,256])
        y_true= tf.reduce_max(tf.one_hot(y_true,4), axis=3)
        #print(y_true.ref().deref())
#         print(y_true[0,0,0,:].numpy().tolist())
#         print(y_true[0,:,:,:])
#         print(tf.reshape(y_true[0,:,:,:],[65536,3]))
#         print(tf.one_hot(tf.reshape(y_true[0,:,:,:],[65536,3]), 4, axis=1))
#         print(tf.reduce_max(tf.one_hot(tf.reshape(y_true[0,:,:,:],[65536,3]), 4), axis=0))
        weights = y_true * class_weights
        #print(weights)
        weights = tf.reduce_sum(weights, axis=3)#
        
        return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)#, weights=weights)

    return weighted_cross_entropy
