from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses

def keras_abs(x):
    return Lambda(lambda x: K.abs(x))(x)

def keras_acos(x):
    return Lambda(lambda x: tf.math.acos(x))(x)

def keras_acosh(x):
    return Lambda(lambda x: tf.math.acosh(x))(x)

def keras_add(x, y):
    x_tf = tf.convert_to_tensor(x.numpy(), dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y.numpy(), dtype=tf.float32)
    return Lambda(lambda x: tf.math.add(*x))([x_tf, y_tf])

def keras_addbmm(A, B, C, beta=1, alpha=1):
    return alpha * tf.linalg.matmul(A, B) + beta * C

def keras_addmm(C, A, B, beta=1.0, alpha=1.0):
    result = tf.linalg.matmul(A, B)
    return beta * C + alpha * result

def keras_addmv(x, y, z, beta=1, alpha=1):
    return beta * x + alpha * K.dot(y, z)

def keras_matmul(x, y, beta=1, alpha=1):
    return Lambda(lambda x: beta * x[0] + alpha * K.dot(x[1], x[2]))([x, y, x])

def keras_matvec(x, y, beta=1, alpha=1):
    return Lambda(lambda x: beta * x[0] + alpha * K.dot(x[1], x[2]))([x, y, x])

def keras_all(x):
    return np.all(x)

def keras_allclose(x, y, rtol=1e-05, atol=1e-08):
    return np.all(np.less_equal(np.abs(x - y), atol + rtol * np.abs(y)))

def keras_amax(x):
    return np.amax(x)

def keras_any(x):
    return tf.reduce_any(x).numpy()

def keras_arange(start, end, step):
    return tf.keras.backend.arange(start, end, step).numpy()

def keras_argmax(x):
    return tf.keras.backend.argmax(x, axis=-1)

def keras_atleast_1d(*tensors):
    return [tf.keras.backend.stack([tensor], axis=0) for tensor in tensors]

def keras_atleast_2d(*tensors):
    return tf.keras.backend.stack([tf.keras.backend.stack([tensor], axis=0) if tf.keras.backend.ndim(tensor) == 1 else tensor for tensor in tensors])

def keras_atleast_3d(*tensors):
    return tf.keras.backend.stack([tf.keras.backend.expand_dims(tensor, axis=[0, 1]) if tf.keras.backend.ndim(tensor) == 1 else tf.keras.backend.expand_dims(tensor, axis=0) if tf.keras.backend.ndim(tensor) == 2 else tensor for tensor in tensors])

def keras_atan(input):
    return tf.math.atan(input)

def keras_atan2(input1, input2):
    return tf.keras.backend.atan2(input1, input2)

def keras_baddbmm(input, batch1, batch2, beta=1, alpha=1):
    return alpha * tf.keras.backend.batch_dot(batch1, batch2) + beta * input

def keras_bartlett_window(window_length, dtype=None):
    return tf.signal.bartlett_window(window_length, dtype=dtype)

def keras_bernoulli(input):
    return tf.keras.backend.random_bernoulli(tf.shape(input), p=input)

def keras_bitwise_right_shift(input, other):
    return tf.bitwise.right_shift(input, other)

def keras_bitwise_and(input1, input2):
    return tf.bitwise.bitwise_and(input1, input2)

def keras_cat(tensors, axis=0):
    tensors = [tf.convert_to_tensor(tensor) for tensor in tensors]
    return tf.keras.backend.concatenate(tensors, axis=axis)

def keras_ceil(input):
    return tf.keras.backend.ceil(input)

def keras_split(input, num_or_size_splits, axis=0):
    return K.tf.split(input, num_or_size_splits=num_or_size_splits, axis=axis)

def keras_clip(input, min, max):
    return K.clip(input, min, max)

def keras_identity(input):
    return K.identity(input)

def keras_concat(tensors, axis=0):
    return K.concatenate(tensors, axis=axis)

def keras_cosh(input):
    return K.cosh(input)

def keras_cos(input):
    return K.cos(input)

def keras_cumprod(input, axis):
    return K.cumprod(input, axis=axis)

def keras_cumsum(input, axis):
    return K.cumsum(input, axis=axis)

def keras_cosh(input):
    return K.cosh(input)

def keras_cos(input):
    return K.cos(input)

def keras_celu(input, alpha):
    return K.celu(input, alpha)

from tensorflow import keras

def nn_keras_global_average_pooling1d(input_tensor):
    return keras.layers.GlobalAveragePooling1D()(input_tensor)

def nn_keras_global_average_pooling2d(input_tensor):
    return keras.layers.GlobalAveragePooling2D()(input_tensor)

def nn_keras_global_average_pooling3d(input_tensor):
    return keras.layers.GlobalAveragePooling3D()(input_tensor)

def nn_keras_global_max_pooling1d(input_tensor):
    return keras.layers.GlobalMaxPooling1D()(input_tensor)

def nn_keras_global_max_pooling2d(input_tensor):
    return keras.layers.GlobalMaxPooling2D()(input_tensor)

def nn_keras_global_max_pooling3d(input_tensor):
    return keras.layers.GlobalMaxPooling3D()(input_tensor)

def nn_keras_alpha_dropout(input_tensor, rate):
    return keras.layers.AlphaDropout(rate)(input_tensor)

def nn_keras_average_pooling1d(input_tensor, pool_size, strides, padding):
    return keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_keras_average_pooling2d(input_tensor, pool_size, strides, padding):
    return keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_keras_average_pooling3d(input_tensor, pool_size, strides, padding):
    return keras.layers.AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_keras_batch_normalization(input_tensor, epsilon, momentum):
    return keras.layers.BatchNormalization(epsilon=epsilon, momentum=momentum)(input_tensor)

def nn_keras_binary_crossentropy(input_tensor, target, from_logits=False):
    return keras.losses.binary_crossentropy(target, input_tensor, from_logits=from_logits)

def nn_keras_celu(input_tensor, alpha):
    return keras.layers.Activation('celu', alpha=alpha)(input_tensor)

def nn_keras_constant_pad1d(input, padding, value):
    padded = tf.pad(input, [[0, 0], padding, [0, 0]], mode='CONSTANT', constant_values=value)
    return padded[:, :, 0]

def nn_keras_constant_pad2d(input, padding, value):
    return keras.layers.ZeroPadding2D(padding)(input) + value

def nn_keras_constant_pad3d(input, padding, value):
    return tf.pad(input, padding, mode='CONSTANT', constant_values=value)

def nn_keras_conv1d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_keras_conv2d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_keras_conv3d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return keras.layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_keras_elu(input, alpha):
    return keras.layers.ELU(alpha)(input)

def nn_keras_cosine_similarity(y_true, y_pred, axis=-1):
    return K.losses.cosine_similarity(y_true, y_pred, axis=axis)

def nn_keras_sparse_categorical_crossentropy(y_true, y_pred, from_logits=False):
    return K.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
    
def nn_keras_embedding_bag(indices, x, W, axis):
    return K.layers.EmbeddingBag(indices, x, W, axis=axis)

def nn_keras_embedding(input, num_embeddings, embedding_dim, padding_idx=None):
    return K.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=(padding_idx is not None), input_length=None)

def nn_keras_feature_alpha_dropout(input, p=0.5):
    return K.layers.AlphaDropout(rate=p)(input)

def nn_keras_flatten(input):
    return K.layers.Flatten()(input)

def nn_keras_fold(output_size, kernel_size, dilation=1, padding=0, stride=1):
    raise NotImplementedError("Keras does not have a direct equivalent of Fold.")

def nn_keras_fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None):
    return K.FractionalMaxPooling2D(pool_size=kernel_size, ratio=output_ratio)(input)

def nn_keras_fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None):
    return K.FractionalMaxPooling3D(pool_size=kernel_size, ratio=output_ratio)(input)

def nn_keras_adaptive_avg_pool1d(input, output_size):
    return K.layers.GlobalAveragePooling1D()(input)

def nn_keras_adaptive_avg_pool2d(input, output_size):
    return K.layers.GlobalAveragePooling2D()(input)

def nn_keras_adaptive_avg_pool3d(input, output_size):
    return K.layers.GlobalAveragePooling3D()(input)

def keras_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return layers.LayerNormalization(epsilon=eps)(input)

def keras_leaky_relu(x, alpha=0.01):
    return layers.LeakyReLU(alpha=alpha)(x)

def keras_linear(input, weight, bias=None):
    return layers.Dense(units=weight.shape[0], use_bias=(bias is not None), kernel_initializer=tf.keras.initializers.Constant(weight), bias_initializer=tf.keras.initializers.Constant(bias))(input)

def keras_local_response_norm(input, size=5, alpha=0.0001, beta=0.75, k=1.0):
    return layers.LocallyConnected1D(filters=input.shape[-1], kernel_size=size, padding='same')(input)

def keras_logsigmoid(input):
    return tf.math.log_sigmoid(input)

def keras_log_softmax(input, axis=None):
    return layers.Softmax(axis=axis)(input)

def keras_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return layers.MaxPooling1D(pool_size=kernel_size, strides=stride, padding='valid')(input)

def keras_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return layers.MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=stride, padding='valid')(input)

def keras_margin_ranking_loss(input1, input2, target, margin=0, reduction='mean'):
    return tf.keras.losses.MarginRankingLoss(margin=margin, reduction=reduction)(target, input1, input2)

def keras_max_pool1d(input, pool_size, strides=None, padding='valid'):
    return layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(input)

def keras_max_pool2d(input, pool_size, strides=None, padding='valid'):
    return layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(input)

def keras_max_pool3d(input, pool_size, strides=None, padding='valid'):
    return layers.MaxPooling3D(pool_size=pool_size, strides=strides, padding=padding)(input)

def keras_max_unpool1d(input, size, strides=None, padding='valid'):
    return tf.image.resize(input, size=size, method='nearest')

def keras_max_unpool2d(input, size, strides=None, padding='valid'):
    return tf.image.resize(input, size=size, method='nearest')

def keras_max_unpool3d(input, size, strides=None, padding='valid'):
    return tf.image.resize(input, size=size, method='nearest')

def keras_mish(input):
    return input * tf.math.tanh(tf.math.softplus(input))

def keras_mse_loss(input, target, reduction='mean'):
    return losses.MeanSquaredError(reduction=reduction)(target, input)

def keras_multilabel_margin_loss(input, target, reduction='mean'):
    return losses.binary_crossentropy(target, input, from_logits=True, reduction=reduction)

def keras_multilabel_soft_margin_loss(input, target, reduction='mean'):
    return losses.BinaryCrossentropy(from_logits=True, reduction=reduction)(target, input)
