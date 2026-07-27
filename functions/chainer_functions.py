import chainer
import chainer.functions as F
import numpy as np
import chainer.links as L

def chainer_absolute(x):
    if not isinstance(x, chainer.Variable):
        x = np.array(x, dtype=np.float32)  
        x = chainer.Variable(x)            
    return F.absolute(x)

def chainer_acos(x):
    if isinstance(x, chainer.Variable):
        x = x.array
    x = F.clip(x, -1.0, 1.0)
    return F.arccos(x).data

def chainer_acosh(x):
    if isinstance(x, chainer.Variable):
        x = x.array
    x = np.array(x, dtype=np.float32)
    x = chainer.Variable(x)
    return F.log(x + F.sqrt(x**2 - 1))

def chainer_add(x, y):
    x_np = np.array(x)
    y_np = np.array(y)
    x_var = chainer.Variable(x_np)
    y_var = chainer.Variable(y_np)
    return F.add(x_var, y_var)

def chainer_arccosh(x):
    return F.log1p(x + (x**2 - 1)**0.5)


def safe_divide(x1, x2, epsilon=1e-8):
    x1_data = x1.data if isinstance(x1, chainer.Variable) else x1
    x2_data = x2.data if isinstance(x2, chainer.Variable) else x2
    
    safe_x2 = x2_data + (np.isclose(x2_data, 0.0) * epsilon)
    safe_division_result = x1_data / safe_x2
    
    if isinstance(x1, chainer.Variable):
        return chainer.as_variable(safe_division_result.astype(x1_data.dtype))
    elif isinstance(x2, chainer.Variable):
        return chainer.as_variable(safe_division_result.astype(x2_data.dtype))
    else:
        return safe_division_result

def chainer_addcdiv(t, x1, x2, value=1):
    safe_division_result = safe_divide(x1, x2)
    return t + value * safe_division_result

def chainer_addcmul(input_array, tensor1, tensor2, value=1.0):
    input_array = np.asarray(input_array, dtype=np.float32)
    tensor1 = np.asarray(tensor1, dtype=np.float32)
    tensor2 = np.asarray(tensor2, dtype=np.float32)

    if input_array.ndim != 1 or tensor1.ndim != 1 or tensor2.ndim != 1:
        raise ValueError("All inputs must be 1D arrays")
    if len(input_array) != len(tensor1) or len(tensor1) != len(tensor2):
        raise ValueError("All inputs must be the same length")

    product = np.multiply(tensor1, tensor2)
    result = input_array + value * product

    return result

def chainer_addmv(x, y, z, beta=1, alpha=1):
    return beta * x + alpha * F.matmul(y, z)

def chainer_addmm(A, B, C):
    return F.matmul(A, B) + C

def chainer_all(x):
    return np.all(x)

def chainer_amin(x):
    return chainer.functions.min(x).array

def chainer_arcsinh(x):
    return F.arcsinh(np.array([x], dtype=np.float32)).array[0]

def chainer_argmax(x):
    return np.argmax(x.numpy())

def chainer_argmin(x):
    return np.argmin(x)

def chainer_atleast_1d(*tensors):
    return [chainer.Variable(tensor) for tensor in tensors]

def chainer_atleast_2d(*tensors):
    return chainer.Variable(np.stack([np.atleast_2d(tensor) for tensor in tensors]))


def chainer_atleast_3d(*tensors):
    return chainer.Variable(np.stack([np.atleast_3d(tensor) for tensor in tensors]))

def chainer_atan(input):
    return F.arctan(input)

def chainer_atan2(input1, input2):
    return F.atan2(input1, input2).data

def chainer_atanh(input):
    return F.arctanh(input).data

def chainer_baddbmm(input, batch1, batch2, beta=1, alpha=1):
    return alpha * F.batch_matmul(batch1, batch2).data + beta * input

def chainer_bartlett_window(window_length, dtype=None):
    return F.bartlett(window_length, dtype=dtype).array

def chainer_bernoulli(input):
    return F.bernoulli(input)

def chainer_empty_bfloat16(size):
    return F.empty((size,), dtype=np.float16)

def chainer_bincount(input, weights=None, minlength=0):
    if weights is not None:
        weights = np.array(weights, dtype=np.float32)
    input = np.array(input, dtype=np.int32)
    bins = np.arange(minlength)
    hist, _ = np.histogram(input, bins=bins, weights=weights)
    return hist


def chainer_bitwise_left_shift(input, other):
    input = np.array(input, dtype=np.int32)
    return input << other


def chainer_bitwise_not(input):
    return ~np.array(input, dtype=np.int32)

def chainer_bitwise_or(input1, input2):
    return np.bitwise_or(np.array(input1, dtype=np.int32), np.array(input2, dtype=np.int32))

def chainer_bitwise_right_shift(input, other):
    return np.right_shift(np.array(input, dtype=np.int32), other)

def chainer_bitwise_xor(input1, input2):
    return np.bitwise_xor(np.array(input1, dtype=np.int32), np.array(input2, dtype=np.int32))

def chainer_bitwise_and(input1, input2):
    return np.bitwise_and(np.array(input1, dtype=np.int32), np.array(input2, dtype=np.int32))

def chainer_bmm(input, mat2):
    xp = chainer.backend.get_array_module(input)
    input_ch = chainer.Variable(xp.array(input))
    mat2_ch = chainer.Variable(xp.array(mat2))
    return chainer.functions.matmul(input_ch, mat2_ch)

def chainer_broadcast_tensors(*tensors):
    return chainer.functions.broadcast(*[np.array(tensor) for tensor in tensors])

def chainer_broadcast_shapes(shape1, shape2):
    shape1 = np.array(shape1)
    shape2 = np.array(shape2)
    result_shape = np.broadcast(shape1, shape2).shape
    return result_shape
    
def chainer_broadcast_to(input_tensor, size):
    return F.broadcast_to(input_tensor, size)


def chainer_bucketize(input_tensor, boundaries):
    boundaries = np.array(boundaries)
    return np.digitize(input_tensor, bins=boundaries)

def chainer_cat(tensors, axis=0):
    return F.concat(tensors, axis=axis)

def chainer_cdist(x1, x2, p=2):
    return F.pairwise_distance(x1, x2, metric='euclidean' if p == 2 else 'minkowski', p=p)

def chainer_ceil(input):
    return F.ceil(input)

def chainer_chain_matmul(*matrices):
    return F.matmul(*matrices)

def chainer_cholesky_inverse(input, upper=False):
    L = F.cholesky(input)
    L_inv = np.linalg.inv(L.array)
    return L_inv @ L_inv.T if not upper else L_inv.T @ L_inv

def chainer_cholesky(input):
    return F.cholesky(input)

def chainer_cholesky_solve(input1, input2, upper=False):
    return F.cholesky_solve(input1, input2, upper=upper)

def chainer_split_axis(input, indices_or_sections, axis=0):
    return F.split_axis(input, indices_or_sections=indices_or_sections, axis=axis)

def chainer_clip(input, min, max):
    return F.clip(input, min, max)
    
def chainer_stack(tensors, axis=1):
    return F.stack(tensors, axis=axis)

def chainer_complex(real, imag):
    return F.complex(real, imag)

def chainer_concat(tensors, axis=0):
    tensors = [np.array(tensor) for tensor in tensors]
    return F.concat(tensors, axis=axis)

def chainer_conj(input):
    input_np = input.array if isinstance(input, chainer.Variable) else input
    conj_result = np.conj(input_np)
    
    if isinstance(input, chainer.Variable):
        return chainer.Variable(conj_result)
    
    return conj_result

def chainer_copysign(input, other):
    return F.copysign(input, other)

def chainer_cosh(input):
    return F.cosh(input)

def chainer_cos(input):
    return F.cos(input)

def chainer_count_nonzero(input, dim=None):
    return F.sum(F.cast(input != 0, dtype='int32'), axis=dim, keepdims=False)

def chainer_cross(input, other, axis=-1):
    return F.cross(input, other, axis=axis)

def chainer_cummax(input, axis):
    return F.cummax(input, axis=axis)

def chainer_cummin(input, axis):
    return F.cummin(input, axis=axis)

def chainer_cumprod(input, axis):
    return F.cumprod(input, axis=axis)

def chainer_cumsum(input, axis):
    return F.cumsum(input, axis=axis)

def chainer_deg2rad(input):
    return F.radians(input)

def chainer_det(input):
    return F.det(input)

def chainer_diag_embed(input, offset=0, dim1=-2, dim2=-1):
    return F.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)

def chainer_diag(input, diagonal=0):
    return F.diagonal(input, offset=diagonal)

def chainer_diagonal(input, offset=0, dim1=0, dim2=1):
    return F.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)

def chainer_digamma(input):
    return F.digamma(input)

def chainer_dist(input, other, p=2):
    return F.norm(input - other, p=p)


def chainer_avg_pool1d(input, kernel_size, stride):
    input_tensor = chainer.Variable(np.array(input, dtype=np.float32))
    return F.average_pooling_1d(input_tensor, ksize=kernel_size, stride=stride)

def chainer_adaptive_avg_pool2d(input, output_size):
    return F.adaptive_avg_pool2d(input, output_size)

def chainer_adaptive_avg_pool3d(input, output_size):
    return F.adaptive_avg_pool3d(input, output_size)


def chainer_adaptive_max_pool1d(input_tensor, output_size):
    if isinstance(output_size, int):
        output_size = (output_size,)

    kernel_size = input_tensor.shape[2] // output_size[0]
    stride = kernel_size
    return F.max_pooling_1d(input_tensor, ksize=kernel_size, stride=stride)

def chainer_adaptive_max_pool2d(input, output_size):
    return F.adaptive_max_pool2d(input, output_size)


def chainer_adaptive_max_pool3d(input_tensor, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)
    if not isinstance(output_size, tuple):
        raise TypeError("output_size must be a tuple")

    kernel_size = (
        input_tensor.shape[2] // output_size[0],
        input_tensor.shape[3] // output_size[1],
        input_tensor.shape[4] // output_size[2]
    )
    return F.max_pooling_nd(input_tensor, ksize=kernel_size, stride=kernel_size)

def chainer_avg_pool1d(input, kernel_size, stride=None, padding=0):
    return F.average_pooling_1d(input, kernel_size, stride=stride, pad=padding)

def chainer_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False):
    return F.average_pooling_2d(input, kernel_size, stride=stride, pad=padding, cover_all=ceil_mode)

def chainer_avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False):
    return F.average_pooling_3d(input, kernel_size, stride=stride, pad=padding, cover_all=ceil_mode)

def chainer_batch_norm(input, num_features, eps=1e-05, decay=0.9, use_global_stats=False):
    return F.batch_normalization(input, eps=eps, decay=decay, use_global_stats=use_global_stats)

def chainer_bce_loss(input, target, weight=None, size_average=None, reduce=None):
    return F.sigmoid_cross_entropy(input, target)

def chainer_bce_with_logits_loss(input, target, weight=None):
    input = input.astype('float32')
    target = target.astype('int32')
    return F.sigmoid_cross_entropy(input, target).array

def chainer_celu(input, alpha=1.0):
    return F.celu(input, alpha=alpha)

def chainer_constant_pad_1d(input, padding, value):
    if isinstance(padding, (float, np.float32, np.float64, chainer.Variable)):
        padding = int(padding)

    pad_width = (padding, padding)
    
    input_np = input.array if isinstance(input, chainer.Variable) else input
    padded_input = np.pad(input_np, pad_width=pad_width, mode='constant', constant_values=value)
    
    return padded_input

def chainer_constant_pad_2d(input, padding, value):
    return F.pad(input, pad_width=[(0, 0), (padding, padding), (padding, padding)], mode='constant', constant_values=value)

def chainer_constant_pad_3d(input, padding, value):
    if isinstance(padding, (float, np.float32, np.float64)):
        padding = int(padding)
    
    pad_width = [(0, 0), (padding, padding), (padding, padding), (padding, padding)]
    
    input_np = input.array if isinstance(input, chainer.Variable) else input
    padded_input = np.pad(input_np, pad_width=pad_width, mode='constant', constant_values=value)
    
    return padded_input

def chainer_conv1d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    if len(input.shape) != 3:
        raise ValueError("Input tensor must be 3D (batch_size, in_channels, length)")

    in_channels = input.shape[1]
    conv1d = L.Convolution1D(in_channels, out_channels, kernel_size, stride=stride, pad=padding, dilate=dilation, groups=groups, nobias=not bias)
    return conv1d(input)

def chainer_conv2d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    in_channels = input.shape[1]
    conv2d = chainer.links.Convolution2D(in_channels, out_channels, kernel_size, stride=stride, pad=padding, dilate=dilation, groups=groups, bias=bias)
    return conv2d(input)

def chainer_conv3d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    if input.ndim != 5:
        raise ValueError("Input tensor must be 5D (batch_size, in_channels, depth, height, width)")
    
    in_channels = input.shape[1]
    conv3d = L.Convolution3D(in_channels, out_channels, kernel_size, stride=stride, pad=padding, dilate=dilation, groups=groups, nobias=not bias)
    return conv3d(input)

def nn_chainer_adaptive_avg_pool1d(input_tensor, output_size):
    return F.adaptive_avg_pool1d(input_tensor, output_size)

def nn_chainer_adaptive_avg_pool2d(input_tensor, output_size):
    return F.adaptive_avg_pool2d(input_tensor, output_size)

def nn_chainer_adaptive_avg_pool3d(input_tensor, output_size):
    return F.adaptive_avg_pool3d(input_tensor, output_size)

def nn_chainer_adaptive_max_pool1d(input_tensor, output_size):
    return F.adaptive_max_pool1d(input_tensor, output_size)

def nn_chainer_adaptive_max_pool2d(input_tensor, output_size):
    return F.adaptive_max_pool2d(input_tensor, output_size)

def nn_chainer_adaptive_max_pool3d(input_tensor, output_size):
    return F.adaptive_max_pool3d(input_tensor, output_size)

def nn_chainer_average_pooling1d(input_tensor, kernel_size, stride, padding):
    return F.average_pooling_1d(input_tensor, kernel_size, stride=stride, pad=padding)

def nn_chainer_average_pooling2d(input_tensor, kernel_size, stride, padding):
    return F.average_pooling_2d(input_tensor, kernel_size, stride=stride, pad=padding)

def nn_chainer_average_pooling3d(input_tensor, kernel_size, stride, padding):
    return F.average_pooling_3d(input_tensor, kernel_size, stride=stride, pad=padding)

def nn_chainer_batch_normalization(input_tensor, num_features, eps, decay):
    bn_layer = L.BatchNormalization(num_features, eps=eps, decay=decay)
    return bn_layer(input_tensor)

def nn_chainer_binary_cross_entropy(input_tensor, target, reduce='mean'):
    return F.binary_cross_entropy(input_tensor, target, reduce=reduce)

def nn_chainer_celu(input_tensor, alpha):
    return F.celu(input_tensor, alpha)

def nn_chainer_constant_pad1d(input, padding, value):
    return F.pad(input, ((0, 0), (0, 0), padding), mode='constant', constant_values=value)

def nn_chainer_constant_pad2d(input, padding, value):
    return F.pad(input, ((0, 0), (0, 0), padding, padding), mode='constant', constant_values=value)

def nn_chainer_constant_pad3d(input, padding, value):
    return F.pad(input, ((0, 0), (0, 0), padding, padding, padding), mode='constant', constant_values=value)

def nn_chainer_conv1d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    conv = L.Convolution1D(None, out_channels, kernel_size, stride, padding, dilation, groups, nobias=not bias)
    return conv(input)

def nn_chainer_conv2d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    conv = L.Convolution2D(None, out_channels, kernel_size, stride, padding, dilation, groups, nobias=not bias)
    return conv(input)

def nn_chainer_conv3d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    conv = L.Convolution3D(None, out_channels, kernel_size, stride, padding, dilation, groups, nobias=not bias)
    return conv(input)

def nn_chainer_elu(input, alpha):
    return F.elu(input, alpha)

def nn_chainer_cosine_embedding_loss(y_true, y_pred, margin=0.0):
    return F.cosine_embedding_loss(y_true, y_pred, margin=margin)

def nn_chainer_cosine_similarity(x1, x2, dim, eps=1e-08):
    return F.cosine_similarity(x1, x2, axis=dim, eps=eps)

def nn_chainer_softmax_cross_entropy(y_pred, y_true, axis):
    return F.softmax_cross_entropy(y_pred, y_true, axis=axis)

def nn_chainer_ctc_loss(y_pred, y_true, use_data_lengths=False, blank_label=0):
    return F.contrib.ctc_loss(y_pred, y_true, use_data_lengths=use_data_lengths, blank_label=blank_label)

def nn_chainer_dropout(input, p):
    return F.dropout(input, ratio=p)

def nn_chainer_elu(input, alpha=1.0):
    return F.elu(input, alpha=alpha)

def chainer_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return F.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

def chainer_leaky_relu_(input, negative_slope=0.01, inplace=False):
    return F.leaky_relu(input, slope=negative_slope)

def chainer_leaky_relu(input, negative_slope=0.01):
    return F.leaky_relu(input, slope=negative_slope)

def chainer_linear(input, weight, bias=None):
    return F.linear(input, weight, bias=bias)

def chainer_local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    return F.local_response_normalization(input, n=size, k=k, alpha=alpha, beta=beta)

def chainer_logsigmoid(input):
    return F.logsigmoid(input)

def chainer_log_softmax(input, axis=None):
    return F.log_softmax(input, axis=axis)

def chainer_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return F.lp_pooling_1d(input, norm_type, kernel_size, stride=stride, use_cudnn=False)

def chainer_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return F.lp_pooling_2d(input, norm_type, kernel_size, stride=stride, use_cudnn=False)

def chainer_margin_ranking_loss(input1, input2, target, margin=0, reduction='mean'):
    return F.margin_ranking_loss(input1, input2, target, margin=margin, reduction=reduction)

def chainer_max_pool1d(input, kernel_size, stride=None, padding=0):
    return F.max_pooling_1d(input, ksize=kernel_size, stride=stride, pad=padding)

def chainer_max_pool2d(input, kernel_size, stride=None, padding=0):
    return F.max_pooling_2d(input, ksize=kernel_size, stride=stride, pad=padding)

def chainer_max_pool3d(input, kernel_size, stride=None, padding=0):
    return F.max_pooling_3d(input, ksize=kernel_size, stride=stride, pad=padding)

def chainer_max_unpool1d(input, indices, kernel_size, stride=None, padding=0):
    return F.unpooling_1d(input, indices, ksize=kernel_size, stride=stride, pad=padding)

def chainer_max_unpool2d(input, indices, kernel_size, stride=None, padding=0):
    return F.unpooling_2d(input, indices, ksize=kernel_size, stride=stride, pad=padding)

def chainer_max_unpool3d(input, indices, kernel_size, stride=None, padding=0):
    return F.unpooling_3d(input, indices, ksize=kernel_size, stride=stride, pad=padding)

def chainer_mish(input):
    return input * F.tanh(F.softplus(input))

def chainer_mse_loss(input, target, reduction='mean'):
    return F.mean_squared_error(input, target)

def chainer_multilabel_margin_loss(input, target, reduction='mean'):
    return F.multi_label_margin_loss(input, target, reduction=reduction)

def chainer_multilabel_soft_margin_loss(input, target, reduction='mean'):
    return F.multi_label_soft_margin_loss(input, target, reduction=reduction)