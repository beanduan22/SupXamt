import torch
import torch.nn.functional as F
import torch.nn as nn

def torch_abs(x):
    return torch.abs(x)

def torch_absolute(x):
    return torch.absolute(x)

def torch_acos(x):
    return torch.acos(x)

def torch_acosh(x):
    return torch.acosh(x)

def torch_add(x, y, alpha=1):
    return torch.add(x, y, alpha=alpha)


def torch_addbmm(A, B, C, beta=1, alpha=1):
    A = A.unsqueeze(0) if A.dim() == 2 else A
    B = B.unsqueeze(0) if B.dim() == 2 else B
    C = C.unsqueeze(0) if C.dim() == 2 else C

    max_batch_size = max(A.size(0), B.size(0), C.size(0))
    A = A.repeat(max_batch_size // A.size(0), 1, 1)
    B = B.repeat(max_batch_size // B.size(0), 1, 1)
    C = C.repeat(max_batch_size // C.size(0), 1, 1)

    return torch.addbmm(C, A, B, beta=beta, alpha=alpha)


def torch_addcdiv(x, tensor1, tensor2, value=1):
    return torch.addcdiv(x, tensor1, tensor2, value=value)

def torch_addcmul(x, tensor1, tensor2, value=1):
    return torch.addcmul(x, tensor1, tensor2, value=value)


def torch_addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    beta = float(beta)
    alpha = float(alpha)
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def torch_addmv(x, y, z, beta=1, alpha=1):
    return torch.addmv(x, y, z, beta=beta, alpha=alpha)

def torch_all(x):
    return torch.all(x).item()

def torch_allclose(x, y, rtol=1e-05, atol=1e-08):
    return torch.allclose(x, y, rtol=rtol, atol=atol).item()

def torch_amax(x):
    return torch.amax(x).item()

def torch_amin(x):
    return torch.amin(x).item()

def torch_angle(x):
    if x.nelement() == 0:
        raise ValueError("Input tensor is empty.")
    angle = torch.angle(x)
    return angle.item() if angle.nelement() == 1 else angle

def torch_any(x):
    return torch.any(x).item()

def torch_arange(start, end, step):
    result = torch.arange(start, end, step)
    if result.device.type == 'cuda':
        result = result.cpu()
    return result.numpy()

def torch_arccosh(x):
    return torch.arccosh(x)

def torch_arcsin(x):
    return torch.arcsin(x)

def torch_arcsinh(x):
    return torch.arcsinh(x)

def torch_arctan(x):
    return torch.arctan(x)

def torch_arctanh(x):
    return torch.arctanh(x)

def torch_argmax(x):
    return torch.argmax(x)

def torch_argmin(x):
    return torch.argmin(x)

def torch_argsort(x):
    return torch.argsort(x)

def torch_as_strided(x, size, stride, storage_offset=0):
    return torch.as_strided(x, size=size, stride=stride, storage_offset=storage_offset)

def torch_as_tensor(data, dtype=torch.int32, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def torch_asinh(x):
    return torch.asinh(x)

def torch_atleast_1d(*tensors):
    return [torch.atleast_1d(tensor) for tensor in tensors]

def torch_atleast_2d(*tensors):
    return torch.stack([torch.atleast_2d(tensor) for tensor in tensors])

def torch_atleast_3d(*tensors):
    return torch.stack([torch.atleast_3d(tensor) for tensor in tensors])

def torch_atan(input):
    return torch.atan(input)

def torch_atan2(input1, input2):
    return torch.atan2(input1, input2)

def torch_atanh(input):
    return torch.atanh(input)

def torch_baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None):
    return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha, out=out)

def torch_bartlett_window(window_length, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    return torch.bartlett_window(window_length, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

def torch_bernoulli(input, *, generator=None, out=None):
    return torch.bernoulli(input, generator=generator, out=out)

def torch_BFloat16Storage(size):
    return torch.BFloat16Storage(size)

def torch_bincount(input, weights=None, minlength=0):
    return torch.bincount(input, weights=weights, minlength=minlength)

def torch_bitwise_left_shift(input, other):
    return torch.bitwise_left_shift(input, other)

def torch_bitwise_not(input):
    return torch.bitwise_not(input)

def torch_bitwise_or(input1, input2):
    return torch.bitwise_or(input1, input2)

def torch_bitwise_right_shift(input, other):
    return torch.bitwise_right_shift(input, other)

def torch_bitwise_xor(input1, input2):
    return torch.bitwise_xor(input1, input2)


def torch_bitwise_and(input1, input2):
    return torch.bitwise_and(input1, input2)


def torch_blackman_window(window_length):
    if isinstance(window_length, float):
        window_length = int(window_length)
    elif isinstance(window_length, torch.Tensor):
        window_length = int(window_length.item())
    
    if not isinstance(window_length, int):
        raise TypeError(f"Expected int for window_length, but got {type(window_length).__name__}")

    return torch.blackman_window(window_length)

def torch_bmm(input, mat2):
    if input.dim() != 3 or mat2.dim() != 3:
        raise ValueError("Both inputs must be 3D tensors.")
    return torch.bmm(input, mat2)

def torch_broadcast_tensors(*tensors):
    return torch.broadcast_tensors(*tensors)

def torch_broadcast_shapes(shape1, shape2):
    if isinstance(shape1, torch.Tensor):
        shape1 = shape1.tolist() if shape1.dim() > 0 else shape1.item()
    if isinstance(shape2, torch.Tensor):
        shape2 = shape2.tolist() if shape2.dim() > 0 else shape2.item()
    
    return torch.broadcast_shapes(shape1, shape2)

def torch_broadcast_to(input_tensor, size):
    return torch.broadcast_to(input_tensor, size)

def torch_bucketize(input_tensor, boundaries):
    boundaries = torch.tensor(boundaries)
    return torch.bucketize(input_tensor, boundaries)

def torch_can_cast(from_dtype):
    if not isinstance(from_dtype, torch.dtype) or not isinstance(to_dtype, torch.dtype):
        raise TypeError("Expected torch.dtype objects for from_dtype and to_dtype.")
    return torch.can_cast(from_dtype)

def torch_cartesian_prod(*tensors):
    return torch.cartesian_prod(*tensors)

def torch_cat(tensors, dim=0):
    return torch.cat(tensors, dim=dim)

def torch_cdist(x1, x2, p=2, compute_mode=None, dist=None, dtype=None):
    return torch.cdist(x1, x2, p=p, compute_mode=compute_mode)

def torch_ceil(input):
    return torch.ceil(input)

def torch_chain_matmul(*matrices):
    return torch.chain_matmul(*matrices)

def torch_char_storage(size):
    return torch.CharStorage(size)

def torch_cholesky_inverse(input, upper=False):
    return torch.cholesky_inverse(input, upper=upper)

def torch_cholesky(input):
    return torch.cholesky(input)

def torch_cholesky_solve(input1, input2, upper=False):
    return torch.cholesky_solve(input1, input2, upper=upper)

def torch_chunk(input, chunks, dim=0):
    return torch.chunk(input, chunks, dim=dim)

def torch_clamp(input, min, max, *, out=None):
    return torch.clamp(input, min, max, out=out)

def torch_clip(input, min, max, *, out=None):
    return torch.clip(input, min, max, out=out)

def torch_clone(input):
    return torch.clone(input)

def torch_column_stack(tensors):
    return torch.column_stack(tensors)

def torch_combinations(iterable, r, *, with_replacement=False):
    return torch.combinations(iterable, r, with_replacement=with_replacement)

def torch_compiled_with_cxx11_abi():
    return torch.compiled_with_cxx11_abi()

def torch_ComplexDoubleStorage(size):
    return torch.ComplexDoubleStorage(size)

def torch_ComplexFloatStorage(size):
    return torch.ComplexFloatStorage(size)

def torch_complex(real, imag, *, out=None):
    return torch.complex(real, imag, out=out)


def torch_concat(tensors, dim=0):
    tensors = [torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
    
    if isinstance(dim, torch.Tensor):
        dim = int(dim.item())
    
    return torch.cat(tensors, dim=dim)

def torch_conj(input):
    return torch.conj(input)

def torch_copysign(input, other):
    return torch.copysign(input, other)

def torch_cosh(input):
    return torch.cosh(input)

def torch_cos(input):
    return torch.cos(input)

def torch_count_nonzero(input, dim=None):
    return torch.count_nonzero(input, dim=dim)

def torch_cross(input, other, dim=-1):
    return torch.cross(input, other, dim=dim)

def torch_cummax(input, dim):
    return torch.cummax(input, dim=dim)

def torch_cummin(input, dim):
    return torch.cummin(input, dim=dim)

def torch_cumprod(input, dim):
    return torch.cumprod(input, dim=dim)

def torch_cumsum(input, dim):
    return torch.cumsum(input, dim=dim)

def torch_dequantize(input, scale, zero_point):
    return torch.dequantize(input, scale=scale, zero_point=zero_point)

def torch_det(input):
    return torch.det(input)

def torch_diag_embed(input, offset=0, dim1=-2, dim2=-1):
    return torch.diag_embed(input, offset=offset, dim1=dim1, dim2=dim2)

def torch_diagflat(input, offset=0):
    return torch.diagflat(input, offset=offset)

def torch_diag(input, diagonal=0, out=None):
    return torch.diag(input, diagonal=diagonal, out=out)

def torch_diagonal(input, offset=0, dim1=0, dim2=1):
    return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)

def torch_diff(input, n=1, axis=-1, prepend=torch.tensor([]), append=torch.tensor([])):
    return torch.diff(input, n=n, dim=axis, prepend=prepend, append=append)

def torch_digamma(input, out=None):
    return torch.digamma(input, out=out)

def torch_dist(input, other, p=2, out=None):
    return torch.dist(input, other, p=p, out=out)

def torch_adaptive_avg_pool1d(input_tensor, output_size):
    output_size = tuple(int(x) for x in output_size)
    return torch.nn.functional.adaptive_avg_pool1d(input_tensor, output_size)

def torch_adaptive_avg_pool2d(input, output_size):
    return F.adaptive_avg_pool2d(input, output_size)

def torch_adaptive_avg_pool3d(input, output_size):
    return F.adaptive_avg_pool3d(input, output_size)

def torch_adaptive_max_pool1d(input_tensor, output_size):
    if isinstance(output_size, int):
        output_size = (output_size,)
    elif isinstance(output_size, torch.Tensor):
        if output_size.dtype in [torch.float32, torch.float64]:
            output_size = tuple(int(x) for x in output_size.tolist())
        else:
            output_size = tuple(output_size.tolist())
    elif not isinstance(output_size, tuple):
        raise TypeError("Output size must be an integer, tuple, or integer tensor")

    return torch.nn.functional.adaptive_max_pool1d(input_tensor, output_size)

def torch_adaptive_max_pool2d(input, output_size):
    return F.adaptive_max_pool2d(input, output_size)


def torch_adaptive_max_pool3d(input_tensor, output_size):
    if isinstance(output_size, (int, float)):
        output_size = (int(output_size),) * 3
    elif isinstance(output_size, torch.Tensor):
        if output_size.dtype in [torch.float32, torch.float64]:
            output_size = tuple(output_size.to(torch.int64).tolist())
        else:
            output_size = tuple(output_size.tolist())
    elif isinstance(output_size, list):
        output_size = tuple(map(int, output_size))
    
    return F.adaptive_max_pool3d(input_tensor, output_size)

def torch_alpha_dropout(input, rate):
    return F.alpha_dropout(input, rate)



def torch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode=False):
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = (kernel_size - 1) // 2
        else:
            raise ValueError(f"Invalid padding value: {padding}")

    input_tensor = torch.tensor(input, dtype=torch.float32)
    return F.avg_pool1d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

def torch_avg_pool2d(input, kernel_size, stride, padding, ceil_mode):
    return F.avg_pool2d(input, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

def torch_avg_pool3d(input, kernel_size, stride, padding, ceil_mode):
    return F.avg_pool3d(input, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

def torch_batch_norm(input, num_features, eps, momentum, track_running_stats):
    return F.batch_norm(input, eps=eps, momentum=momentum, affine=True, track_running_stats=track_running_stats)

def torch_bce_loss(input, target, weight):
    return F.binary_cross_entropy(input, target, weight=weight)

def torch_bce_with_logits_loss(input, target, weight=None):
    return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight)

def torch_celu(input, alpha):
    return F.celu(input, alpha)

def torch_constant_pad_1d(input, padding, value):
    if isinstance(padding, torch.Tensor):
        padding = int(padding.item())
    return torch.nn.functional.pad(input, (padding, padding), mode='constant', value=value)

def torch_constant_pad_2d(input, padding, value):
    return torch.nn.functional.pad(input, (padding, padding, padding, padding), mode='constant', value=value)

def torch_constant_pad_3d(input, padding, value):
    if isinstance(padding, torch.Tensor):
        padding = int(padding.item())
    padding_tuple = (padding, padding, padding, padding, padding, padding)
    
    return torch.nn.functional.pad(input, padding_tuple, mode='constant', value=value)

def torch_conv1d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    if input.dim() != 3:
        raise ValueError("Input tensor must be 3D (batch_size, in_channels, length)")
    
    in_channels = input.shape[1]
    
    conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    return conv1d(input)

def torch_conv2d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    in_channels = input.shape[1]
    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    return conv2d(input)

def torch_conv3d(input, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    if input.dim() != 5:
        raise ValueError("Input tensor must be 5D (batch_size, in_channels, depth, height, width)")
    
    in_channels = input.shape[1]
    
    conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    return conv3d(input)

def nn_torch_adaptive_avg_pool1d(input_tensor, output_size):
    return nn.AdaptiveAvgPool1d(output_size)(input_tensor)

def nn_torch_adaptive_avg_pool2d(input_tensor, output_size):
    return nn.AdaptiveAvgPool2d(output_size)(input_tensor)

def nn_torch_adaptive_avg_pool3d(input_tensor, output_size):
    return nn.AdaptiveAvgPool3d(output_size)(input_tensor)

def nn_torch_adaptive_max_pool1d(input_tensor, output_size):
    return nn.AdaptiveMaxPool1d(output_size)(input_tensor)

def nn_torch_adaptive_max_pool2d(input_tensor, output_size):
    return nn.AdaptiveMaxPool2d(output_size)(input_tensor)

def nn_torch_adaptive_max_pool3d(input_tensor, output_size):
    return nn.AdaptiveMaxPool3d(output_size)(input_tensor)

def nn_torch_alpha_dropout(input_tensor, p):
    return nn.AlphaDropout(p)(input_tensor)

def nn_torch_avg_pool1d(input_tensor, kernel_size, stride, padding, ceil_mode):
    return nn.AvgPool1d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)(input_tensor)

def nn_torch_avg_pool2d(input_tensor, kernel_size, stride, padding, ceil_mode):
    return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)(input_tensor)

def nn_torch_avg_pool3d(input_tensor, kernel_size, stride, padding, ceil_mode):
    return nn.AvgPool3d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)(input_tensor)

def nn_torch_batch_norm1d(input_tensor, num_features, eps, momentum):
    return nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)(input_tensor)

def nn_torch_batch_norm2d(input_tensor, num_features, eps, momentum):
    return nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)(input_tensor)

def nn_torch_batch_norm3d(input_tensor, num_features, eps, momentum):
    return nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)(input_tensor)

def nn_torch_bce_loss(input_tensor, target, weight, size_average, reduce, reduction):
    return nn.BCELoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)(input_tensor, target)

def nn_torch_bce_with_logits_loss(input_tensor, target, weight, size_average, reduce, reduction, pos_weight):
    return nn.BCEWithLogitsLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)(input_tensor, target)

def nn_torch_bilinear(input1, input2, in1_features, in2_features, out_features, bias):
    return nn.Bilinear(in1_features, in2_features, out_features, bias=bias)(input1, input2)

def nn_torch_celu(input_tensor, alpha):
    return nn.CELU(alpha)(input_tensor)

def nn_torch_constant_pad1d(input, padding, value):
    return F.pad(input, padding, 'constant', value)

def nn_torch_constant_pad2d(input, padding, value):
    return F.pad(input, padding, 'constant', value)

def nn_torch_constant_pad3d(input, padding, value):
    return F.pad(input, padding, 'constant', value)

def nn_torch_conv1d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(input.size(1), out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)(input)

def nn_torch_conv2d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(input.size(1), out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)(input)

def nn_torch_conv3d(input, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv3d(input.size(1), out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)(input)

def nn_torch_elu(input, alpha):
    return F.elu(input, alpha)

def nn_torch_conv_transpose1d(input, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
    conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias, dilation=dilation)
    return conv(input)

def nn_torch_conv_transpose2d(input, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias, dilation=dilation)
    return conv(input)

def nn_torch_conv_transpose3d(input, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
    conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias, dilation=dilation)
    return conv(input)

def nn_torch_cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean'):
    loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    return loss_fn(input1, input2, target)

def nn_torch_cosine_similarity(input1, input2, dim=1, eps=1e-08):
    return nn.functional.cosine_similarity(input1, input2, dim=dim, eps=eps)

def nn_torch_cross_entropy_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss_fn(input, target)

def nn_torch_ctc_loss(input, target, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    loss_fn = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    return loss_fn(input, target, input_lengths, target_lengths)

def nn_torch_data_parallel(module, device_ids=None, output_device=None, dim=0):
    return nn.DataParallel(module, device_ids=device_ids, output_device=output_device, dim=dim)

def nn_torch_dropout2d(input, p=0.5, inplace=False):
    return nn.functional.dropout2d(input, p=p, inplace=inplace)

def nn_torch_dropout3d(input, p=0.5, inplace=False):
    return nn.functional.dropout3d(input, p=p, inplace=inplace)

def nn_torch_dropout(input, p=0.5, inplace=False):
    return nn.functional.dropout(input, p=p, inplace=inplace)

def nn_torch_elu(input, alpha=1.0, inplace=False):
    return nn.functional.elu(input, alpha=alpha, inplace=inplace)

def nn_torch_embedding_bag(input, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None, padding_idx=None):
    emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, mode=mode, sparse=sparse, padding_idx=padding_idx)
    return emb_bag(input)

def nn_torch_embedding(input, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
    return nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

def nn_torch_feature_alpha_dropout(input, p=0.5, inplace=False):
    return nn.functional.feature_alpha_dropout(input, p=p, inplace=inplace)

def nn_torch_flatten(input, start_dim=1, end_dim=-1):
    return nn.functional.flatten(input, start_dim=start_dim, end_dim=end_dim)

def nn_torch_fold(output_size, kernel_size, dilation=1, padding=0, stride=1):
    return nn.Fold(output_size, kernel_size, dilation=dilation, padding=padding, stride=stride)

def nn_torch_fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    return nn.FractionalMaxPool2d(kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)

def nn_torch_fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    return nn.FractionalMaxPool3d(kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)

def nn_torch_adaptive_avg_pool1d(input, output_size):
    return nn.functional.adaptive_avg_pool1d(input, output_size)

def nn_torch_adaptive_avg_pool2d(input, output_size):
    return nn.functional.adaptive_avg_pool2d(input, output_size)

def nn_torch_adaptive_avg_pool3d(input, output_size):
    return nn.functional.adaptive_avg_pool3d(input, output_size)

def nn_torch_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

def nn_torch_leaky_relu_(input, negative_slope=0.01, inplace=False):
    return torch.nn.functional.leaky_relu_(input, negative_slope=negative_slope, inplace=inplace)

def nn_torch_leaky_relu(input, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(input, negative_slope=negative_slope)

def nn_torch_linear(input, weight, bias=None):
    return torch.nn.functional.linear(input, weight, bias=bias)

def nn_torch_local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    return torch.nn.functional.local_response_norm(input, size, alpha=alpha, beta=beta, k=k)

def nn_torch_logsigmoid(input):
    return torch.nn.functional.logsigmoid(input)

def nn_torch_log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return torch.nn.functional.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def nn_torch_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return torch.nn.functional.lp_pool1d(input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)

def nn_torch_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)

def nn_torch_margin_ranking_loss(input1, input2, target, margin=0, reduction='mean'):
    return nn.MarginRankingLoss(margin=margin, reduction=reduction)(input1, input2, target)

def nn_torch_max_pool1d(input, kernel_size, stride=None, padding=0):
    return nn.functional.max_pool1d(input, kernel_size, stride=stride, padding=padding)

def nn_torch_max_pool2d(input, kernel_size, stride=None, padding=0):
    return nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding)

def nn_torch_max_pool3d(input, kernel_size, stride=None, padding=0):
    return nn.functional.max_pool3d(input, kernel_size, stride=stride, padding=padding)

def nn_torch_max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return nn.functional.max_unpool1d(input, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)

def nn_torch_max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return nn.functional.max_unpool2d(input, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)

def nn_torch_max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return nn.functional.max_unpool3d(input, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)

def nn_torch_mish(input):
    return input * torch.tanh(nn.functional.softplus(input))

def nn_torch_mse_loss(input, target, reduction='mean'):
    return nn.MSELoss(reduction=reduction)(input, target)

def nn_torch_multilabel_margin_loss(input, target, reduction='mean'):
    return nn.MultiLabelMarginLoss(reduction=reduction)(input, target)

def nn_torch_multilabel_soft_margin_loss(input, target, reduction='mean'):
    return nn.MultiLabelSoftMarginLoss(reduction=reduction)(input, target)