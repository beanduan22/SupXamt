import jax.numpy as jnp
import jax.random as jrandom
import jax
import jax.scipy.special as jsp
from jax import nn

def jax_abs(x):
    return jnp.abs(x)

def jax_acos(x):
    return jnp.arccos(x)

def jax_acosh(x):
    return jnp.arccosh(x)

def jax_add(x, y):
    return jnp.add(x, y)

def jax_addbmm(A, B, C, beta=1, alpha=1):
    return alpha * jnp.matmul(A, B) + beta * C

def jax_addcdiv(t, x, y, value=1):
    x = jnp.array(x)
    y = jnp.array(y)
    t = jnp.array(t)

    division_result = jnp.divide(x, y)
    scaled_division = value * division_result
    return t + scaled_division

def jax_addcmul(input, tensor1, tensor2, value=1):
    product = jnp.multiply(tensor1, tensor2)
    return input + value * product


def jax_addmm(C, A, B, beta=1.0, alpha=1.0):
    result = jnp.dot(A, B)
    return beta * C + alpha * result


def jax_addmv(x, y, z, beta=1, alpha=1):
    return beta * x + alpha * jnp.dot(y, z)

def jax_all(x):
    return jnp.all(x).item()


def jax_allclose(x, y, rtol=1e-05, atol=1e-08):
    return jnp.all(jnp.less_equal(jnp.abs(x - y), atol + rtol * jnp.abs(y))).item()


def jax_amax(x):
    return jnp.amax(x).item()

def jax_amin(x):
    return jnp.min(x).item()


def jax_angle(x):
    if x.size == 0:
        raise ValueError("Input array is empty.")
    angle = jnp.angle(x)
    return angle.item() if angle.size == 1 else angle

def jax_any(x):
    return jnp.any(x).item()

def jax_arange(start, end, step):
    return jnp.arange(start, end, step).tolist()

def jnp_arccosh(x):
    return jnp.arccosh(x)

def jax_arcsin(x):
    return jnp.arcsin(x)


def jax_arcsinh(x):
    return jnp.arcsinh(x)

def jax_arctan(x):
    return jnp.arctan(x)

def jax_arctanh(x):
    return jnp.arctanh(x)

def jax_argmax(x):
    return jnp.argmax(x)

def jax_argmin(x):
    return jnp.argmin(x)

def jax_argsort(x):
    return jnp.argsort(x)

def jax_asinh(x):
    return jnp.arcsinh(x)

def jax_atleast_1d(*tensors):
    return [jnp.atleast_1d(tensor) for tensor in tensors]

def jax_atleast_2d(*tensors):
    return jnp.stack([jnp.atleast_2d(tensor) for tensor in tensors])

def jax_atleast_3d(*tensors):
    return jnp.stack([jnp.atleast_3d(tensor) for tensor in tensors])

def jax_atan(input):
    return jnp.arctan(input)

def jax_atan2(input1, input2):
    return jnp.arctan2(input1, input2)

def jax_atanh(input):
    return jnp.arctanh(input)

def jax_baddbmm(input, batch1, batch2, beta=1, alpha=1):
    return alpha * jnp.matmul(batch1, batch2) + beta * input

def jax_bartlett_window(window_length, dtype=None):
    return jnp.bartlett(window_length)

def jax_bernoulli(input, key):
    return jrandom.bernoulli(key, p=input)

def jax_bitwise_right_shift(input, other):
    return jnp.right_shift(input, other)


def jax_bitwise_xor(input1, input2):
    return jnp.bitwise_xor(input1, input2)

def jax_bitwise_and(input1, input2):
    return jnp.bitwise_and(input1, input2)

def jax_blackman_window(window_length):
    return jnp.blackman(window_length)

def jax_cat(tensors, axis=0):
    tensors = [jnp.array(tensor) for tensor in tensors]
    return jnp.concatenate(tensors, axis=axis)

def jax_cdist(x1, x2, p=2):
    return jnp.linalg.norm(x1[:, None] - x2[None, :], ord=p, axis=-1)

def jax_ceil(input):
    return jnp.ceil(input)

def jax_chain_matmul(*matrices):
    return jnp.linalg.multi_dot(matrices)

def jax_complex(real, imag):
    return jnp.complex(real, imag)

def jax_concat(tensors, axis=0):
    return jnp.concatenate(tensors, axis=axis)

def jax_conj(input):
    return jnp.conj(input)

def jax_cosh(input):
    return jnp.cosh(input)

def jax_cos(input):
    return jnp.cos(input)

def jax_deg2rad(input):
    return jnp.radians(input)

def jax_digamma(input):
    return jsp.digamma(input)

def jax_dist(input, other, p=2):
    return jnp.linalg.norm(input - other, ord=p)

def jax_celu(input, alpha):
    return jax.nn.celu(input, alpha)

def nn_jax_celu(input_tensor, alpha):
    def celu(x, alpha):
        return jnp.maximum(0, x) + jnp.minimum(0, alpha * (jnp.exp(x / alpha) - 1))
    return celu(input_tensor, alpha)

def nn_jax_constant_pad1d(input, padding, value):
    return jnp.pad(input, ((0, 0), (0, 0), padding), mode='constant', constant_values=value)

def nn_jax_constant_pad2d(input, padding, value):
    return jnp.pad(input, ((0, 0), (0, 0), padding, padding), mode='constant', constant_values=value)

def nn_jax_constant_pad3d(input, padding, value):
    return jnp.pad(input, ((0, 0), (0, 0), padding, padding, padding), mode='constant', constant_values=value)

def nn_jax_conv1d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.conv(input, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

def nn_jax_conv2d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.conv(input, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

def nn_jax_elu(input, alpha):
    return nn.elu(input, alpha)

def nn_jax_cosine_similarity(x1, x2, axis):
    return nn.cosine_similarity(x1, x2, axis=axis)

def nn_jax_softmax_cross_entropy(logits, labels):
    return nn.softmax_cross_entropy(logits, labels)

def nn_jax_dropout(x, p):
    return nn.dropout(x, rate=p)

def nn_jax_elu(x, alpha=1.0):
    return nn.elu(x, alpha=alpha)

def jax_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    raise NotImplementedError("JAX layer_norm is not implemented.")

def jax_leaky_relu(input, negative_slope=0.01):
    return jax.nn.leaky_relu(input, negative_slope=negative_slope)

def jax_linear(input, weight, bias=None):
    return jnp.dot(input, weight) + (bias if bias is not None else 0)

def jax_local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    raise NotImplementedError("JAX local_response_norm is not implemented.")

def jax_logsigmoid(input):
    return jax.nn.logsigmoid(input)

def jax_log_softmax(input, axis=None):
    return jax.nn.log_softmax(input, axis=axis)

def jax_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    raise NotImplementedError("JAX lp_pool1d is not implemented.")

def jax_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    raise NotImplementedError("JAX lp_pool2d is not implemented.")
def jax_margin_ranking_loss(input1, input2, target, margin=0.0):
    return jax.numpy.mean(jnp.maximum(0, margin - target * (input1 - input2)))

def jax_max_pool1d(input, kernel_size, stride=None, padding=0):
    return jax.lax.max_pool(input, window_shape=(kernel_size,), strides=(stride,), padding=padding)

def jax_max_pool2d(input, kernel_size, stride=None, padding=0):
    return jax.lax.max_pool(input, window_shape=(kernel_size, kernel_size), strides=(stride, stride), padding=padding)

def jax_max_pool3d(input, kernel_size, stride=None, padding=0):
    return jax.lax.max_pool(input, window_shape=(kernel_size, kernel_size, kernel_size), strides=(stride, stride, stride), padding=padding)

def jax_max_unpool1d(input, indices, kernel_size, stride=None, padding=0):
    return jax.lax.upsample(input, size=(kernel_size,), strides=(stride,), padding=padding)

def jax_max_unpool2d(input, indices, kernel_size, stride=None, padding=0):
    return jax.lax.upsample(input, size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding)

def jax_max_unpool3d(input, indices, kernel_size, stride=None, padding=0):
    return jax.lax.upsample(input, size=(kernel_size, kernel_size, kernel_size), strides=(stride, stride, stride), padding=padding)

def jax_mish(input):
    return input * jnp.tanh(jax.nn.softplus(input))

def jax_mse_loss(input, target, reduction='mean'):
    return jnp.mean(jnp.square(input - target))

def jax_multilabel_margin_loss(input, target, reduction='mean'):
    return jnp.mean(jnp.maximum(0, 1 + target * (input - target)))

def jax_multilabel_soft_margin_loss(input, target, reduction='mean'):
    return jnp.mean(jnp.log1p(jnp.exp(-input * target)))