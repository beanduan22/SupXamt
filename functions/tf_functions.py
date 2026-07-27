import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses

def tf_abs(x):
    return tf.abs(x)

def tf_acos(x):
    return tf.acos(x)

def tf_acosh(x):
    return tf.acosh(x)

def tf_add(x, y):
    return tf.add(x, y)

def tf_addbmm(A, B, C, beta=1, alpha=1):
    return alpha * tf.linalg.matmul(A, B) + beta * C

def tf_addcdiv(x, y, z, value=1.0):
    return x + value * (y / z)

def tf_addcmul(input, tensor1, tensor2, value=1):
    product = tf.math.multiply(tensor1, tensor2)
    scaled_product = tf.multiply(product, value)
    return input + scaled_product


def tf_addmm(C, A, B, beta=1.0, alpha=1.0):
    if isinstance(A, (float, int)):
        A = tf.constant([[A]])
    if isinstance(B, (float, int)):
        B = tf.constant([[B]])
    result = tf.linalg.matmul(A, B)
    return beta * C + alpha * result


def tf_addmv(x, y, z, beta=1, alpha=1):
    return beta * x + alpha * tf.linalg.matvec(y, z)

def tf_all(x):
    return tf.reduce_all(x).numpy()

def tf_allclose(x, y, rtol=1e-05, atol=1e-08):
    return tf.reduce_all(tf.math.less_equal(tf.abs(x - y), atol + rtol * tf.abs(y))).numpy()

def tf_amax(x):
    return tf.reduce_max(x).numpy()

def tf_amin(x):
    return tf.reduce_min(x).numpy()

def tf_any(x):
    return tf.reduce_any(x).numpy()

def tf_arange(start, end, step):
    return tf.range(start, end, delta=step).numpy()

def tf_arccosh(x):
    return tf.math.acosh(x)

def tf_arcsin(x):
    return tf.math.asin(x)


def tf_arcsinh(x):
    return tf.math.asinh(x)

def tf_arctan(x):
    return tf.math.atan(x)

def tf_arctanh(x):
    return tf.math.atanh(x)

def tf_argmax(x):
    return tf.argmax(tf.convert_to_tensor(x), axis=-1, output_type=tf.dtypes.int64)

def tf_argmin(x):
    return tf.argmin(x)

def tf_argsort(x):
    return tf.argsort(x)

def tf_strided_slice(x, size, stride):
    begin = [0] * len(x.shape)
    end = [(s - 1) * st + 1 for s, st in zip(size, stride)]
    return tf.strided_slice(x, begin=begin, end=end, strides=stride)

def tf_constant(data):
    int_data = [int(x) for x in data]
    return tf.constant(int_data, dtype=tf.int32)

def tf_asinh(x):
    return tf.math.asinh(x)

def tf_atleast_1d(*tensors):
    return [tf.convert_to_tensor(tensor) for tensor in tensors]

def tf_atleast_2d(*tensors):
    return tf.stack([tf.convert_to_tensor(tf.expand_dims(tensor, axis=0) if tf.rank(tensor) == 1 else tensor) for tensor in tensors])

def tf_atleast_3d(*tensors):
    return tf.stack([tf.expand_dims(tensor, axis=[0, 1]) if tf.rank(tensor) == 1 else tf.expand_dims(tensor, axis=0) if tf.rank(tensor) == 2 else tensor for tensor in tensors])

def tf_atan(input):
    return tf.math.atan(input)

def tf_atan2(input1, input2):
    return tf.math.atan2(input1, input2)

def tf_atanh(input):
    return tf.math.atanh(input)

def tf_baddbmm(input, batch1, batch2, beta=1, alpha=1):
    return tf.matmul(batch1, batch2) * alpha + input * beta

def tf_bartlett_window(window_length, dtype=tf.float32):
    return 1 - tf.abs((2 * tf.range(window_length, dtype=dtype) / (window_length - 1)) - 1)

def tf_bernoulli(input):
    return tf.random.uniform(tf.shape(input)) < input

def tf_bincount(input, weights=None, minlength=0):
    return tf.math.bincount(input, weights=weights, minlength=minlength)

def tf_bitwise_left_shift(input, other):
    return tf.bitwise.left_shift(input, other)

def tf_bitwise_not(input):
    return tf.bitwise.invert(input)


def tf_bitwise_right_shift(input, other):
    return tf.bitwise.right_shift(input, other)

def tf_bitwise_xor(input1, input2):
    return tf.bitwise.bitwise_xor(input1, input2)

def keras_bitwise_xor(input1, input2):
    return tf.bitwise.bitwise_xor(input1, input2)


def tf_bitwise_and(input1, input2):
    return tf.bitwise.bitwise_and(input1, input2)

def tf_blackman_window(window_length):
    window = np.blackman(window_length)
    return tf.convert_to_tensor(window, dtype=tf.float32)

def tf_bmm(input, mat2):
    return tf.linalg.matmul(input, mat2)

def tf_broadcast_tensors(*tensors):
    return [tf.broadcast_to(tensor, tf.broadcast_dynamic_shape(tf.shape(tensor), tf.shape(tensors[-1]))) for tensor in tensors]

def tf_broadcast_shapes(shape1, shape2):
    return tf.broadcast_static_shape(tf.TensorShape(shape1), tf.TensorShape(shape2)).as_list()

def tf_broadcast_to(input_tensor, size):
    return tf.broadcast_to(input_tensor, size)

def tf_bucketize(input_tensor, boundaries):
    boundaries = boundaries if isinstance(boundaries, (list, np.ndarray)) else boundaries.numpy().tolist()
    return tf.raw_ops.Bucketize(input=input_tensor, boundaries=boundaries)

def tf_can_cast(from_dtype, to_dtype):
    try:
        temp_tensor = tf.constant(0, dtype=from_dtype)
        cast_tensor = tf.cast(temp_tensor, dtype=to_dtype)
        return True
    except TypeError:
        return False

def tf_cat(tensors, axis=0):
    return tf.concat(tensors, axis=axis)

def tf_cdist(x1, x2, p=2):
    return tf.norm(x1[:, None] - x2[None, :], ord=p, axis=-1)

def tf_ceil(input):
    return tf.math.ceil(input)

def tf_chain_matmul(*matrices):
    return tf.linalg.matmul(*matrices)

def tf_cholesky_inverse(input, upper=False):
    cholesky_decomp = tf.linalg.cholesky(input)
    if upper:
        cholesky_decomp = tf.transpose(cholesky_decomp)
    return tf.linalg.cholesky_solve(cholesky_decomp, tf.eye(input.shape[-1]))

def tf_cholesky(input):
    return tf.linalg.cholesky(input)

def tf_cholesky_solve(input1, input2, upper=False):
    return tf.linalg.cholesky_solve(tf.linalg.cholesky(input1), input2, adjoint=upper)

def tf_split(input, num_or_size_splits, axis=0):
    return tf.split(input, num_or_size_splits=num_or_size_splits, axis=axis)

def tf_clip_by_value(input, clip_value_min, clip_value_max):
    return tf.clip_by_value(input, clip_value_min=clip_value_min, clip_value_max=clip_value_max)

def tf_identity(input):
    return tf.identity(input)

def tf_stack(tensors, axis=1):
    return tf.stack(tensors, axis=axis)

def tf_complex(real, imag):
    return tf.complex(real, imag)

def tf_concat(tensors, axis=0):
    tensors = [tf.convert_to_tensor(tensor) for tensor in tensors]
    return tf.concat(tensors, axis=axis)

def tf_conj(input):
    return tf.math.conj(input)

def tf_copysign(input, other):
    return tf.multiply(tf.sign(other), tf.abs(input))

def tf_cosh(input):
    return tf.math.cosh(input)

def tf_cos(input):
    return tf.math.cos(input)

def tf_count_nonzero(input, axis=None):
    return tf.math.count_nonzero(input, axis=axis, keepdims=False, dtype=tf.int64)

def tf_cross(input, other):
    return tf.linalg.cross(input, other)

def tf_cummax(input, axis):
    return tf.math.cummax(input, axis=axis, exclusive=False)

def tf_cummin(input, axis):
    return tf.math.cummin(input, axis=axis, exclusive=False)

def tf_cumprod(input, axis):
    return tf.math.cumprod(input, axis=axis)

def tf_cumsum(input, axis):
    return tf.math.cumsum(input, axis=axis)

def tf_deg2rad(input):
    return tf.math.radians(input)

def tf_det(input):
    return tf.linalg.det(input)

def tf_diag_embed(input, offset=0):
    return tf.linalg.diag_embed(input, k=offset)

def tf_diagflat(input, offset=0):
    return tf.linalg.diagflat(input, k=offset)

def tf_diag(input, diagonal=0):
    return tf.linalg.diag_part(input, k=diagonal)

def tf_diagonal(input, offset=0, dim1=0, dim2=1):
    return tf.linalg.diag_part(input, k=offset)

def tf_diff(input, n=1, axis=-1):
    return tf.math.diff(input, n=n, axis=axis)

def tf_digamma(input):
    return tf.math.digamma(input)

def tf_dist(input, other, p=2):
    return tf.norm(input - other, ord=p)

def tf_adaptive_avg_pool1d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling1D()(input)

def tf_adaptive_avg_pool2d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling2D()(input)

def tf_adaptive_avg_pool3d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling3D()(input)

def tf_adaptive_max_pool1d(input_data):
    input_tensor, output_size = input_data
    if input_tensor.ndim == 2:
        input_tensor = input_tensor[:, None, :]
    return tf.keras.layers.GlobalMaxPooling1D()(input_tensor)

def tf_adaptive_max_pool2d(input, output_size):
    return tf.keras.layers.GlobalMaxPooling2D()(input)

def tf_adaptive_max_pool3d(input, output_size):
    return tf.keras.layers.GlobalMaxPooling3D()(input)

def tf_alpha_dropout(input, rate):
    return tf.keras.layers.AlphaDropout(rate=rate)(input)

def tf_avg_pool1d(input, pool_size, strides, padding='valid'):
    return tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)(input)

def tf_avg_pool2d(input, pool_size, strides=None, padding='valid'):
    return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(input)

def tf_avg_pool3d(input, pool_size, strides=None, padding='valid'):
    return tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding)(input)

def tf_batch_norm(input):
    return tf.keras.layers.BatchNormalization()(input)

def tf_bce_loss(target, input):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='auto')(target, input)

def tf_bce_with_logits_loss(target, input):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(target, input)

def tf_celu(input):
    return tf.keras.layers.Activation('celu')(input)

def tf_constant_pad_1d(input, padding, value):
    return tf.pad(input, paddings=[[padding, padding]], mode='CONSTANT', constant_values=value)

def tf_constant_pad_2d(input, padding, value):
    return tf.pad(input, paddings=[[padding, padding], [padding, padding]], mode='CONSTANT', constant_values=value)

def tf_constant_pad_3d(input, padding, value):
    if isinstance(padding, tf.Tensor):
        padding = int(padding.numpy())
    paddings = [[padding, padding], [padding, padding], [padding, padding]]
    
    return tf.pad(input, paddings=paddings, mode='CONSTANT', constant_values=value)


def tf_conv1d(input, out_channels, kernel_size, stride=1, padding='valid', dilation=1, groups=1, bias=True):
    if isinstance(padding, int):
        padding = 'valid'
    conv1d = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding.lower(), dilation_rate=dilation, groups=groups, use_bias=bias)
    return conv1d(input)

def tf_conv2d(input, out_channels, kernel_size, stride=1, padding='valid', dilation=1, groups=1, bias=True):
    conv2d = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, groups=groups, use_bias=bias)
    return conv2d(input)


def tf_conv3d(input, out_channels, kernel_size, stride=1, padding='valid', dilation=1, groups=1, bias=True):
    in_channels = input.shape[-1]
    if in_channels % groups != 0:
        groups = 1
    
    conv3d = tf.keras.layers.Conv3D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding.lower(), dilation_rate=dilation, groups=groups, use_bias=bias)
    return conv3d(input)

def nn_tf_global_average_pooling1d(input_tensor):
    return tf.keras.layers.GlobalAveragePooling1D()(input_tensor)

def nn_tf_global_average_pooling2d(input_tensor):
    return tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

def nn_tf_global_average_pooling3d(input_tensor):
    return tf.keras.layers.GlobalAveragePooling3D()(input_tensor)

def nn_tf_global_max_pooling1d(input_tensor):
    return tf.keras.layers.GlobalMaxPooling1D()(input_tensor)

def nn_tf_global_max_pooling2d(input_tensor):
    return tf.keras.layers.GlobalMaxPooling2D()(input_tensor)

def nn_tf_global_max_pooling3d(input_tensor):
    return tf.keras.layers.GlobalMaxPooling3D()(input_tensor)

def nn_tf_alpha_dropout(input_tensor, rate):
    return tf.keras.layers.AlphaDropout(rate)(input_tensor)

def nn_tf_average_pooling1d(input_tensor, pool_size, strides, padding):
    return tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_tf_average_pooling2d(input_tensor, pool_size, strides, padding):
    return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_tf_average_pooling3d(input_tensor, pool_size, strides, padding):
    return tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def nn_tf_batch_normalization(input_tensor, epsilon, momentum):
    return tf.keras.layers.BatchNormalization(epsilon=epsilon, momentum=momentum)(input_tensor)

def nn_tf_binary_crossentropy(input_tensor, target, from_logits=False):
    return tf.keras.losses.binary_crossentropy(target, input_tensor, from_logits=from_logits)

def nn_tf_celu(input_tensor, alpha):
    return tf.keras.layers.Activation(lambda x: tf.nn.crelu(x, alpha))(input_tensor)

def nn_tf_constant_pad1d(input, padding, value):
    padded = tf.pad(input, [[0, 0], padding, [0, 0]], mode='CONSTANT', constant_values=value)
    return padded[:, :, 0]

def nn_tf_constant_pad2d(input, padding, value):
    return tf.pad(input, padding, mode='CONSTANT', constant_values=value)

def nn_tf_constant_pad3d(input, padding, value):
    return tf.pad(input, padding, mode='CONSTANT', constant_values=value)

def nn_tf_conv1d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_tf_conv2d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_tf_conv3d(input, filters, kernel_size, strides, padding, dilation_rate, groups, use_bias):
    return tf.keras.layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias)(input)

def nn_tf_elu(input, alpha):
    return tf.keras.layers.ELU(alpha)(input)

def nn_tf_conv_transpose1d(input, filters, kernel_size, strides=1, padding='valid', output_padding=None, dilation_rate=1, groups=1, use_bias=True):
    return tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, dilation_rate=dilation_rate, use_bias=use_bias)(input)

def nn_tf_conv_transpose2d(input, filters, kernel_size, strides=1, padding='valid', output_padding=None, dilation_rate=1, groups=1, use_bias=True):
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, dilation_rate=dilation_rate, use_bias=use_bias)(input)

def nn_tf_conv_transpose3d(input, filters, kernel_size, strides=1, padding='valid', output_padding=None, dilation_rate=1, groups=1, use_bias=True):
    return tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, dilation_rate=dilation_rate, use_bias=use_bias)(input)

def nn_tf_cosine_similarity(y_true, y_pred, axis=-1, reduction='auto'):
    return tf.keras.losses.CosineSimilarity(axis=axis, reduction=reduction)(y_true, y_pred)

def nn_tf_sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, reduction='auto'):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction=reduction)(y_true, y_pred)

def nn_tf_ctc_loss(y_true, y_pred, input_length, label_length):
    return tf.keras.losses.CTCLoss()(y_true, y_pred, input_length=input_length, label_length=label_length)

def nn_tf_mirrored_strategy(device_ids):
    strategy = tf.distribute.MirroredStrategy(devices=device_ids)
    return strategy

def nn_tf_dropout(input, rate=0.5):
    return tf.keras.layers.Dropout(rate=rate)(input)

def nn_tf_elu(input, alpha=1.0):
    return tf.keras.layers.ELU(alpha=alpha)(input)

def nn_tf_embedding_lookup_sparse(params, sp_ids, sp_weights, combiner='mean'):
    return tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, combiner=combiner)

def nn_tf_embedding(input, num_embeddings, embedding_dim, padding_idx=None):
    return tf.keras.layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=(padding_idx is not None), input_length=None)

def nn_tf_feature_alpha_dropout(input, p=0.5):
    return tf.keras.layers.AlphaDropout(rate=p)(input)

def nn_tf_flatten(input):
    return tf.keras.layers.Flatten()(input)

def nn_tf_fold(output_size, kernel_size, dilation=1, padding=0, stride=1):
    raise NotImplementedError("TensorFlow does not have a direct equivalent of Fold.")

def nn_tf_fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None):
    return tf.nn.fractional_max_pool2d(input, ksize=kernel_size, output_shape=output_size, pooling_ratio=output_ratio)

def nn_tf_fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None):
    return tf.nn.fractional_max_pool3d(input, ksize=kernel_size, output_shape=output_size, pooling_ratio=output_ratio)

def nn_tf_adaptive_avg_pool1d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling1D()(input)

def nn_tf_adaptive_avg_pool2d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling2D()(input)

def nn_tf_adaptive_avg_pool3d(input, output_size):
    return tf.keras.layers.GlobalAveragePooling3D()(input)

def tf_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return tf.keras.layers.LayerNormalization(epsilon=eps, scale=True, center=True)(input)

def tf_leaky_relu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha=alpha)

def tf_linear(input, weight, bias=None):
    return tf.matmul(input, weight) + (bias if bias is not None else 0)

def tf_local_response_normalization(input, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75):
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

def tf_logsigmoid(x):
    return tf.math.log_sigmoid(x)

def tf_log_softmax(x, axis=None):
    return tf.nn.log_softmax(x, axis=axis)

def tf_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return tf.nn.pool(input, window_shape=[kernel_size], strides=[stride or 1], padding='VALID', pooling_type='MAX')

def tf_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return tf.nn.pool(input, window_shape=[kernel_size, kernel_size], strides=[stride or 1, stride or 1], padding='VALID', pooling_type='MAX')

def tf_margin_ranking_loss(input1, input2, target, margin=0, reduction='mean'):
    return tf.nn.margin_ranking_loss(input1, input2, target, margin=margin, reduction=reduction)

def tf_max_pool1d(input, kernel_size, stride=None, padding='VALID'):
    return tf.nn.max_pool1d(input, ksize=kernel_size, strides=stride, padding=padding)

def tf_max_pool2d(input, kernel_size, stride=None, padding='VALID'):
    return tf.nn.max_pool2d(input, ksize=kernel_size, strides=stride, padding=padding)

def tf_max_pool3d(input, kernel_size, stride=None, padding='VALID'):
    return tf.nn.max_pool3d(input, ksize=kernel_size, strides=stride, padding=padding)

def tf_max_unpool1d(input, indices, kernel_size, stride=None, padding='VALID'):
    return tf.image.extract_patches(input, sizes=[1, kernel_size, 1, 1], strides=[1, stride, 1, 1], rates=[1, 1, 1, 1], padding=padding)

def tf_max_unpool2d(input, indices, kernel_size, stride=None, padding='VALID'):
    return tf.image.extract_patches(input, sizes=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding=padding)

def tf_max_unpool3d(input, indices, kernel_size, stride=None, padding='VALID'):
    return tf.image.extract_patches(input, sizes=[1, kernel_size, kernel_size, kernel_size, 1], strides=[1, stride, stride, stride, 1], rates=[1, 1, 1, 1, 1], padding=padding)

def tf_mish(input):
    return input * tf.math.tanh(tf.math.softplus(input))

def tf_mse_loss(input, target, reduction='mean'):
    return losses.MeanSquaredError(reduction=reduction)(input, target)

def tf_multilabel_margin_loss(input, target, reduction='mean'):
    return losses.binary_crossentropy(target, input, from_logits=True, reduction=reduction)

def tf_multilabel_soft_margin_loss(input, target, reduction='mean'):
    return losses.BinaryCrossentropy(from_logits=True, reduction=reduction)(target, input)
