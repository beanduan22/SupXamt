import hypothesis.strategies as st
import numpy as np
import random
import torch
import tensorflow as tf
import jax.numpy as jnp
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, tuples

def generate_random_tensor(shape):
    return np.random.rand(*shape)

def mutate_tensor(tensor):
    mutation_type = random.choice(['add_noise', 'zero_elements', 'scale'])
    if mutation_type == 'add_noise':
        noise = np.random.normal(0, 0.1, tensor.shape)
        return tensor + noise
    elif mutation_type == 'zero_elements':
        mask = np.random.rand(*tensor.shape) > 0.5
        tensor[mask] = 0
        return tensor
    elif mutation_type == 'scale':
        factor = np.random.uniform(0.5, 1.5)
        return tensor * factor

tensor_strategy = st.builds(generate_random_tensor, st.tuples(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10)))

boundary_tensors = [
    np.zeros((1, 1)),
    np.ones((1, 1)) * np.finfo(np.float32).max,
    np.ones((1, 1)) * np.finfo(np.float32).min,
    np.ones((1, 1)) * np.finfo(np.float32).eps,
]

def generate_test_inputs():
    def ordinary_condition():
        return generate_random_tensor((np.random.randint(1, 10), np.random.randint(1, 10)))

    ordinary_tensors = [ordinary_condition() for _ in range(64)]
    mutated_tensors = [mutate_tensor(tensor) for tensor in ordinary_tensors if random.random() < 0.2]
    
    all_tensors = ordinary_tensors + mutated_tensors + boundary_tensors
    
    return st.sampled_from(all_tensors)

def acos_input_strategy():
    valid_floats = st.floats(min_value=-1.0, max_value=1.0, width=32)
    return st.lists(valid_floats, min_size=1, max_size=10).map(lambda lst: torch.tensor(lst))


def acosh_input_strategy():
    regular_floats = st.floats(min_value=1.0, max_value=1000.0, width=32)
    return st.one_of(regular_floats).map(lambda x: torch.tensor([x]))

def add_input_strategy():
    regular_floats = st.floats(min_value=-1e20, max_value=1e20, width=64)
    special_floats = st.just(float('nan')) | st.just(float('inf')) | st.just(float('-inf'))
    return st.tuples(
        st.one_of(regular_floats, special_floats),
        st.one_of(regular_floats, special_floats)
    ).map(lambda x: (torch.tensor([x[0]]), torch.tensor([x[1]])))

def addbmm_input_strategy():
    float_strategy = st.floats(min_value=-10, max_value=10, allow_infinity=False, allow_nan=False, width=32)
    return st.tuples(
        st.arrays(dtype=np.float32, shape=(1, 3, 3), elements=float_strategy),
        st.arrays(dtype=np.float32, shape=(1, 3, 3), elements=float_strategy),
        st.arrays(dtype=np.float32, shape=(1, 3, 3), elements=float_strategy)
    ).map(lambda x: (torch.tensor(x[0]), torch.tensor(x[1]), torch.tensor(x[2])))
    
def addcdiv_input_strategy():
    regular_floats = st.floats(min_value=1e-5, max_value=1e5, allow_nan=False)
    value_floats = st.floats(min_value=0.1, max_value=10, allow_nan=False)
    return st.tuples(
        st.one_of(regular_floats),
        st.one_of(regular_floats),
        st.one_of(regular_floats),
        st.one_of(value_floats)
    ).map(lambda x: (torch.tensor([x[0]]), torch.tensor([x[1]]), torch.tensor([x[2]]), x[3]))

def addbmm_input_strategy():
    float_strategy = st.floats(min_value=-10, max_value=10, allow_infinity=False, allow_nan=False, width=32)
    return st.tuples(
        arrays(dtype=np.float32, shape=(3, 3), elements=float_strategy),
        arrays(dtype=np.float32, shape=(3, 3, 3), elements=float_strategy),
        arrays(dtype=np.float32, shape=(3, 3, 3), elements=float_strategy)
    ).map(lambda x: (torch.tensor(x[0]), torch.tensor(x[1]), torch.tensor(x[2])))
    
def addcmul_input_strategy():
    regular_floats = st.floats(min_value=-10, max_value=10, width=32)
    return st.tuples(
        st.lists(regular_floats, min_size=5, max_size=5).map(lambda x: torch.tensor(x, dtype=torch.float32)),
        st.lists(regular_floats, min_size=5, max_size=5).map(lambda x: torch.tensor(x, dtype=torch.float32)),
        st.lists(regular_floats, min_size=5, max_size=5).map(lambda x: torch.tensor(x, dtype=torch.float32)),
        st.floats(min_value=0.1, max_value=10)
    )
    
def generate_addmm_inputs():
    float_strategy = st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False)
    return st.tuples(
        float_strategy,
        float_strategy,
        st.lists(st.lists(float_strategy, min_size=4, max_size=4), min_size=3, max_size=3).map(np.array),
        st.lists(st.lists(float_strategy, min_size=5, max_size=5), min_size=4, max_size=4).map(np.array)
    ).map(lambda x: (
        x[0],
        x[1],
        torch.tensor(x[2], dtype=torch.float32),
        torch.tensor(x[3], dtype=torch.float32)
    ))
    
def generate_addmv_inputs():
    float_strategy = st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False)
    return st.tuples(
        st.lists(float_strategy, min_size=3, max_size=3).map(np.array),
        st.lists(st.lists(float_strategy, min_size=3, max_size=3), min_size=3, max_size=3).map(np.array),
        st.lists(float_strategy, min_size=3, max_size=3).map(np.array)
    ).map(lambda x: (torch.tensor(x[0], dtype=torch.float32), torch.tensor(x[1], dtype=torch.float32), torch.tensor(x[2], dtype=torch.float32)))
    
def generate_all_inputs():
    bool_strategy = st.booleans()
    return st.lists(bool_strategy, min_size=1, max_size=100).map(lambda x: (
        torch.tensor(x, dtype=torch.bool),
        np.array(x, dtype=bool),
        tf.constant(x, dtype=tf.bool),
        jnp.array(x, dtype=bool)
    ))
    
def generate_allclose_inputs():
    float_strategy = st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False)
    return st.tuples(float_strategy, float_strategy).map(lambda x: (
        torch.tensor(x[0], dtype=torch.float32),
        torch.tensor(x[1], dtype=torch.float32),
        np.array(x[0], dtype=np.float32),
        np.array(x[1], dtype=np.float32),
        tf.constant(x[0], dtype=tf.float32),
        tf.constant(x[1], dtype=tf.float32),
        jnp.array(x[0], dtype=jnp.float32),
        jnp.array(x[1], dtype=jnp.float32)
    ))
    
def generate_amax_inputs():
    float_list_strategy = st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1)
    return float_list_strategy.map(lambda x: (
        torch.tensor(x, dtype=torch.float32),
        np.array(x, dtype=np.float32),
        tf.constant(x, dtype=tf.float32),
        jnp.array(x, dtype=jnp.float32)
    ))
    
def generate_amin_inputs():
    float_strategy = st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False)
    return float_strategy.map(lambda x: (
        torch.tensor(x, dtype=torch.float32),
        np.array(x, dtype=np.float32),
        tf.constant(x, dtype=tf.float32),
        jnp.array(x, dtype=jnp.float32)
    ))
    
def generate_angle_inputs():
    return st.tuples(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    ).map(lambda real_imag: (
        torch.tensor(complex(real_imag[0], real_imag[1]), dtype=torch.cfloat),
    ))
    
def generate_any_inputs():
    bool_strategy = st.booleans()
    return bool_strategy.map(lambda x: (
        torch.tensor(x),
    ))
    
def generate_arange_inputs():
    start_strategy = st.integers(min_value=0, max_value=100)
    end_strategy = st.integers(min_value=1, max_value=101)
    step_strategy = st.integers(min_value=1, max_value=10)
    return st.tuples(start_strategy, end_strategy, step_strategy)

def generate_arccosh_inputs():
    valid_floats = st.floats(min_value=1.0, max_value=10.0)
    return valid_floats.map(lambda x: torch.tensor(x))

def generate_arcsin_inputs():
    valid_floats = st.floats(min_value=-1.0, max_value=1.0)
    return valid_floats.map(lambda x: torch.tensor(x))


def generate_arcsinh_inputs():
    valid_floats = st.floats(min_value=-10.0, max_value=10.0)
    return valid_floats.map(lambda x: torch.tensor(x, dtype=torch.float32))

def generate_arctan_inputs():
    valid_floats = st.floats(min_value=-1.0, max_value=1.0)
    return valid_floats.map(lambda x: torch.tensor(x, dtype=torch.float32))

def generate_arctanh_inputs():
    valid_floats = st.floats(min_value=-1.0, max_value=1.0)
    return valid_floats.map(lambda x: torch.tensor(x, dtype=torch.float32))

def generate_argmax_inputs():
    return st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=1).map(lambda lst: torch.tensor(lst, dtype=torch.float32))

def generate_argmin_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x, dtype=torch.float32))
def generate_argsort_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x, dtype=torch.float32))


def generate_as_strided_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=10, max_size=100).map(lambda x: torch.tensor(x, dtype=torch.float32)),
        st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4).map(tuple),
        st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4).map(tuple),
        st.integers(min_value=0, max_value=10)
    )
    
def generate_as_tensor_inputs():
    return st.lists(st.integers(min_value=-1000, max_value=1000)).map(lambda x: torch.tensor(x, dtype=torch.int32))

def generate_asinh_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(lambda x: torch.tensor(x, dtype=torch.float32))

def generate_atleast_1d_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x, dtype=torch.float32))

def generate_atleast_2d_inputs():
    return st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2), min_size=2).map(lambda x: [torch.tensor(y) for y in x])

def generate_atleast_3d_inputs():
    return st.lists(st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2), min_size=2), min_size=2).map(lambda x: [torch.tensor(y) for y in x])

def generate_atan_inputs():
    return st.floats(min_value=-1000.0, max_value=1000.0).map(lambda x: torch.tensor(x))

def generate_atan2_inputs():
    return st.tuples(
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=-1000.0, max_value=1000.0)
    ).map(lambda x: (torch.tensor(x[0]), torch.tensor(x[1])))
    
def generate_atanh_inputs():
    return st.floats(min_value=-0.99, max_value=0.99).map(lambda x: torch.tensor(x))

def generate_baddbmm_inputs():
    def tensors():
        input_tensor = st.floats(min_value=-1e6, max_value=1e6).map(lambda x: torch.tensor(x))
        batch1 = st.lists(st.lists(st.floats(min_value=-1e2, max_value=1e2), min_size=1, max_size=2), min_size=1).map(lambda x: torch.tensor(x))
        batch2 = st.lists(st.lists(st.floats(min_value=-1e2, max_value=1e2), min_size=1), min_size=1).map(lambda x: torch.tensor(x))
        return st.tuples(input_tensor, batch1, batch2)

    return tensors()

def generate_bartlett_window_inputs():
    window_length_strategy = st.integers(min_value=1, max_value=100)
    dtype_strategy = st.sampled_from([torch.float32, torch.float64])
    return st.tuples(window_length_strategy, dtype_strategy)

def generate_bernoulli_inputs():
    input_strategy = st.floats(min_value=0.0, max_value=1.0)
    return st.tuples(input_strategy)

def generate_BFloat16Storage_inputs():
    size_strategy = st.integers(min_value=1, max_value=100)
    return size_strategy


def generate_bincount_inputs():
    input_strategy = st.lists(st.integers(min_value=0), min_size=1)
    weights_strategy = st.lists(st.floats(min_value=0), min_size=1)
    minlength_strategy = st.integers(min_value=0)
    return st.tuples(input_strategy, weights_strategy, minlength_strategy)

def generate_bitwise_left_shift_inputs():
    input_strategy = st.lists(st.integers(min_value=0, max_value=1000), min_size=1)
    other_strategy = st.integers(min_value=0, max_value=32)
    return st.tuples(input_strategy, other_strategy)

def generate_bitwise_not_inputs():
    return st.lists(st.integers(min_value=0, max_value=1000), min_size=1)

def generate_bitwise_or_inputs():
    return st.tuples(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1),
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1)
    )
    
def generate_bitwise_right_shift_inputs():
    return st.tuples(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1),
        st.integers(min_value=0, max_value=31)
    )
    
def generate_bitwise_xor_inputs():
    return st.tuples(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1),
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1)
    )
    
def generate_bitwise_and_inputs():
    return st.tuples(
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1),
        st.lists(st.integers(min_value=0, max_value=1000), min_size=1)
    )
    
def generate_blackman_window_inputs():
    return st.integers(min_value=1, max_value=100)

def generate_bmm_inputs():
    return st.tuples(
        st.lists(st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2, max_size=2), min_size=2, max_size=2), min_size=1, max_size=1),
        st.lists(st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2, max_size=2), min_size=2, max_size=2), min_size=1, max_size=1)
    )

def generate_broadcast_tensors_inputs():
    return st.lists(
        st.lists(st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False), min_size=1),
        min_size=1,
        max_size=3
    )
    
def generate_broadcast_shapes_inputs():
    base_shape_strategy = st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5)
    
    def compatible_shape(base_shape):
        return [st.one_of(st.just(1), st.just(int(dim))) for dim in base_shape]

    shape1_strategy = base_shape_strategy.map(lambda base: [int(dim) for dim in base])
    shape2_strategy = base_shape_strategy.flatmap(lambda base: st.tuples(*compatible_shape(base)).map(list))

    return st.tuples(shape1_strategy, shape2_strategy)
    
def generate_broadcast_to_inputs():
    tensor_strategy = st.lists(st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False), min_size=1)
    size_strategy = st.lists(st.integers(min_value=1, max_value=100), min_size=1)
    return st.tuples(tensor_strategy, size_strategy)

def generate_bucketize_inputs():
    input_strategy = st.lists(st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False), min_size=1)
    boundaries_strategy = st.lists(st.floats(min_value=-1e38, max_value=1e38, allow_nan=False, allow_infinity=False), min_size=1).map(sorted)
    return st.tuples(input_strategy, boundaries_strategy)

def generate_can_cast_inputs():
    from_dtype = st.sampled_from([torch.float32, torch.int32])
    return st.tuples(from_dtype)

def generate_can_cast_inputs():
    from_dtype = st.sampled_from([torch.float32, torch.int32])
    to_dtype = st.sampled_from([torch.float64, torch.int64])
    return st.tuples(from_dtype, to_dtype)

def generate_cartesian_prod_inputs():
    return st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: tuple(torch.tensor(i) for i in x))

def generate_cat_inputs():
    return st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: [torch.tensor(i) for i in x])

def generate_cdist_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor)
    )

def generate_ceil_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)

def generate_chain_matmul_inputs():
    return st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: [torch.tensor(i) for i in x])

def generate_char_storage_inputs():
    return st.integers(min_value=1, max_value=100)


def generate_cholesky_inverse_inputs():
    def create_positive_definite_matrix(size):
        A = np.random.rand(size, size)
        return np.dot(A, A.T)

    matrix_size_strategy = st.integers(min_value=2, max_value=10)
    return matrix_size_strategy.map(lambda size: torch.tensor(create_positive_definite_matrix(size), dtype=torch.float32))

def generate_cholesky_inputs():
    def create_positive_definite_matrix(size):
        A = np.random.rand(size, size)
        return np.dot(A, A.T)

    matrix_size_strategy = st.integers(min_value=2, max_value=10)
    return matrix_size_strategy.map(lambda size: torch.tensor(create_positive_definite_matrix(size), dtype=torch.float32))

def generate_cholesky_solve_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=1, max_value=1e6), min_size=1).map(torch.tensor),
        st.lists(st.floats(min_value=1, max_value=1e6), min_size=1).map(torch.tensor)
    )
    
def generate_chunk_inputs():
    tensor_strategy = st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x))
    chunks_strategy = st.integers(min_value=1, max_value=10)
    dim_strategy = st.integers(min_value=0, max_value=1)
    return st.tuples(tensor_strategy, chunks_strategy, dim_strategy)

def generate_clamp_inputs():
    tensor_strategy = st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x))
    min_strategy = st.floats(min_value=-1e6, max_value=0)
    max_strategy = st.floats(min_value=0, max_value=1e6)
    return st.tuples(tensor_strategy, min_strategy, max_strategy)

def generate_clip_inputs():
    return generate_clamp_inputs()

def generate_clone_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x))

def generate_column_stack_inputs():
    return st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: [torch.tensor(t) for t in x])

def generate_combinations_inputs():
    iterable_strategy = st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1)
    r_strategy = st.integers(min_value=1, max_value=5)
    with_replacement_strategy = st.booleans()
    return st.tuples(iterable_strategy, r_strategy, with_replacement_strategy)

def generate_complex_inputs():
    real_strategy = st.floats(min_value=-1e6, max_value=1e6)
    imag_strategy = st.floats(min_value=-1e6, max_value=1e6)
    return st.tuples(real_strategy, imag_strategy)

def generate_concat_inputs():
    return st.tuples(
        st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=2).map(lambda x: [np.array(i) for i in x]),
        st.integers(min_value=0, max_value=1)
    )

def generate_conj_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)

def generate_conj_physical_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)

def generate_copysign_inputs():
    return st.tuples(
        st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor),
        st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)
    )

def generate_corrcoef_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2).map(torch.tensor)

def generate_cosh_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)

def generate_cos_inputs():
    return st.floats(min_value=-1e6, max_value=1e6).map(torch.tensor)

def generate_count_nonzero_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=0, max_value=1)
    )

def generate_cov_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2).map(torch.tensor)

def generate_cross_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=-1, max_value=1)
    )
    
def generate_cummax_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=0, max_value=10)
    )

def generate_cummin_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=0, max_value=10)
    )

def generate_cumprod_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=0, max_value=10)
    )

def generate_cumsum_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=0, max_value=10)
    )

def generate_dequantize_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.floats(min_value=0.1, max_value=10.0),
        st.integers(min_value=0, max_value=255)
    )

def generate_det_inputs():
    return st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor)

def generate_diag_embed_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=-10, max_value=10),
        st.integers(min_value=-10, max_value=10),
        st.integers(min_value=-10, max_value=10)
    )

def generate_diagflat_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=-10, max_value=10)
    )
    
def generate_diag_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=-10, max_value=10)
    )

def generate_diagonal_inputs():
    return st.tuples(
        st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(torch.tensor),
        st.integers(min_value=-10, max_value=10),
        st.integers(min_value=0, max_value=1),
        st.integers(min_value=0, max_value=1)
    )

def generate_diff_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2).map(torch.tensor),
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=-1, max_value=1)
    )

def generate_digamma_inputs():
    return st.lists(st.floats(min_value=0.1, max_value=1e6), min_size=1).map(torch.tensor)

def generate_dist_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=1, max_value=10)
    )
    
def generate_adaptive_avg_pool1d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e3, max_value=1e3), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0).unsqueeze(-1)),
        st.tuples(st.integers(min_value=1, max_value=100).filter(lambda x: x > 0))
    )

def generate_adaptive_avg_pool2d_inputs():
    return st.tuples(
        st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=100)
    )

def generate_adaptive_avg_pool3d_inputs():
    return st.tuples(
        st.lists(st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=100)
    )

def generate_adaptive_max_pool1d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0).unsqueeze(-1)),
        st.integers(min_value=1, max_value=100).map(lambda x: (x,))
    )

def generate_adaptive_max_pool2d_inputs():
    return st.tuples(
        st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=100)
    )

def generate_adaptive_max_pool3d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=125, max_size=125)
            .map(lambda x: torch.tensor(x).view(1, 1, 5, 5, 5)),
        st.integers(min_value=1, max_value=10).map(lambda x: (x, x, x))
    )
    
def generate_alpha_dropout_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.floats(min_value=0, max_value=1)
    )

def generate_avg_pool1d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=10).map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)),
        st.integers(min_value=1, max_value=5).map(int),
        st.integers(min_value=1, max_value=5).map(int),
        st.sampled_from(['valid', 'same'])
    )

def generate_avg_pool2d_inputs():
    return st.tuples(
        st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=0, max_value=1)
    )

def generate_avg_pool3d_inputs():
    return st.tuples(
        st.lists(st.lists(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1), min_size=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=0, max_value=1)
    )

def generate_batch_norm_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=10)
    )

def generate_bce_loss_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=0, max_value=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.lists(st.floats(min_value=0, max_value=1), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.sampled_from([None, torch.tensor([0.5])])
    )

def generate_bce_with_logits_loss_inputs():
    input_strategy = st.lists(st.floats(min_value=0, max_value=1), min_size=1).map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    target_strategy = st.lists(st.floats(min_value=0, max_value=1), min_size=1).map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    
    return st.tuples(input_strategy, target_strategy)


def generate_bilinear_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(lambda x: torch.tensor(x).unsqueeze(0)),
        st.integers(min_value=1, max_value=10)
    )

def generate_celu_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.floats(min_value=0.1, max_value=10)
    )
    
def generate_constant_pad_1d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=1, max_value=100),
        st.floats(min_value=-1e6, max_value=1e6)
    )

def generate_constant_pad_2d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=1, max_value=100),
        st.floats(min_value=-1e6, max_value=1e6)
    )

def generate_constant_pad_3d_inputs():
    def create_3d_tensor():
        depth = np.random.randint(1, 10)
        height = np.random.randint(1, 10)
        width = np.random.randint(1, 10)
        return torch.rand((depth, height, width))

    tensor_strategy = st.just(create_3d_tensor())
    padding_strategy = st.integers(min_value=1, max_value=3)
    value_strategy = st.floats(min_value=-1e6, max_value=1e6)

    return st.tuples(tensor_strategy, padding_strategy, value_strategy)

def generate_conv1d_inputs():
    def create_3d_tensor():
        batch_size = torch.randint(1, 5, (1,)).item()
        in_channels = torch.randint(1, 5, (1,)).item()
        length = torch.randint(10, 100, (1,)).item()
        return torch.rand((batch_size, in_channels, length))

    tensor_strategy = st.just(create_3d_tensor())
    out_channels_strategy = st.integers(min_value=1, max_value=10)
    kernel_size_strategy = st.integers(min_value=1, max_value=10)
    stride_strategy = st.integers(min_value=1, max_value=10)
    padding_strategy = st.integers(min_value=0, max_value=10)
    dilation_strategy = st.integers(min_value=1, max_value=10)
    groups_strategy = st.integers(min_value=1, max_value=5)
    bias_strategy = st.booleans()

    return st.tuples(tensor_strategy, out_channels_strategy, kernel_size_strategy, stride_strategy, padding_strategy, dilation_strategy, groups_strategy, bias_strategy)

def generate_conv2d_inputs():
    return st.tuples(
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1).map(torch.tensor),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=0, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.booleans()
    )

def generate_conv3d_inputs():
    def create_5d_tensor():
        batch_size = torch.randint(1, 5, (1,)).item()
        in_channels = torch.randint(1, 5, (1,)).item()
        depth = torch.randint(10, 20, (1,)).item()
        height = torch.randint(10, 20, (1,)).item()
        width = torch.randint(10, 20, (1,)).item()
        return torch.rand((batch_size, in_channels, depth, height, width))

    tensor_strategy = st.just(create_5d_tensor())
    out_channels_strategy = st.integers(min_value=1, max_value=10)
    kernel_size_strategy = st.integers(min_value=1, max_value=5)
    stride_strategy = st.integers(min_value=1, max_value=5)
    padding_strategy = st.integers(min_value=0, max_value=5)
    dilation_strategy = st.integers(min_value=1, max_value=5)
    groups_strategy = st.integers(min_value=1, max_value=3)
    bias_strategy = st.booleans()

    return st.tuples(tensor_strategy, out_channels_strategy, kernel_size_strategy, stride_strategy, padding_strategy, dilation_strategy, groups_strategy, bias_strategy)

import numpy as np

np.random.seed(0)

def generate_nn_adaptive_avg_pool1d_input(batch_size=1, channels=3, length=64):
    return np.random.rand(batch_size, channels, length).astype(np.float32)

def generate_nn_adaptive_avg_pool2d_input(batch_size=1, channels=3, height=64, width=64):
    return np.random.rand(batch_size, channels, height, width).astype(np.float32)

def generate_nn_adaptive_avg_pool3d_input(batch_size=1, channels=3, depth=64, height=64, width=64):
    return np.random.rand(batch_size, channels, depth, height, width).astype(np.float32)

def generate_nn_alpha_dropout_input(batch_size=1, features=256):
    return np.random.rand(batch_size, features).astype(np.float32)

def generate_nn_avg_pool1d_input(batch_size=1, channels=3, length=64):
    return np.random.rand(batch_size, channels, length).astype(np.float32)

def generate_nn_avg_pool2d_input(batch_size=1, channels=3, height=64, width=64):
    return np.random.rand(batch_size, channels, height, width).astype(np.float32)

def generate_nn_avg_pool3d_input(batch_size=1, channels=3, depth=64, height=64, width=64):
    return np.random.rand(batch_size, channels, depth, height, width).astype(np.float32)

def generate_nn_batch_norm1d_input(num_features, batch_size=1, length=64):
    return np.random.rand(batch_size, num_features, length).astype(np.float32)

def generate_nn_batch_norm2d_input(num_features, batch_size=1, height=64, width=64):
    return np.random.rand(batch_size, num_features, height, width).astype(np.float32)

def generate_nn_batch_norm3d_input(num_features, batch_size=1, depth=64, height=64, width=64):
    return np.random.rand(batch_size, num_features, depth, height, width).astype(np.float32)

def generate_nn_bce_loss_input(batch_size=1, features=256):
    inputs = np.random.rand(batch_size, features).astype(np.float32)
    targets = np.random.randint(0, 2, (batch_size, features)).astype(np.float32)
    return inputs, targets

def generate_nn_bce_with_logits_loss_input(batch_size=1, features=256):
    inputs = np.random.rand(batch_size, features).astype(np.float32)
    targets = np.random.randint(0, 2, size=(batch_size, features)).astype(np.float32)
    weights = np.ones((batch_size, features), dtype=np.float32)
    return inputs, targets, weights

def generate_nn_bilinear_input(batch_size=1, in1_features=256, in2_features=256, out_features=128):
    input1 = np.random.rand(batch_size, in1_features).astype(np.float32)
    input2 = np.random.rand(batch_size, in2_features).astype(np.float32)
    return input1, input2, out_features

def generate_nn_celu_input(batch_size=1, features=256):
    return np.random.rand(batch_size, features).astype(np.float32), 1.0

def generate_nn_constant_pad1d_input():
    return np.random.rand(1, 3, 64).astype(np.float32), (1, 1), 0.0

def generate_nn_constant_pad2d_input():
    return np.random.rand(1, 3, 64, 64).astype(np.float32), ((1, 1), (1, 1)), 0.0

def generate_nn_constant_pad3d_input():
    return np.random.rand(1, 3, 64, 64, 64).astype(np.float32), ((1, 1), (1, 1), (1, 1)), 0.0

def generate_nn_conv1d_input():
    return np.random.rand(1, 3, 64).astype(np.float32), 3, 1, 0, 1, 1, 1, True

def generate_nn_conv2d_input():
    return np.random.rand(1, 3, 64, 64).astype(np.float32), 3, (3, 3), (1, 1), (0, 0), (1, 1), 1, True

def generate_nn_conv3d_input():
    return np.random.rand(1, 3, 64, 64, 64).astype(np.float32), 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, True

def generate_nn_cosine_similarity_input():
    return np.random.rand(10, 3).astype(np.float32), np.random.rand(10, 3).astype(np.float32), 1, 1e-08

def generate_nn_cross_entropy_loss_input():
    return np.random.rand(10, 3).astype(np.float32), np.random.randint(0, 3, size=(10,)).astype(np.int32)

def generate_nn_dropout_input():
    return np.random.rand(1, 10).astype(np.float32), 0.5

def generate_nn_elu_input():
    return np.random.rand(1, 10).astype(np.float32), 1.0

def generate_nn_conv_transpose1d_input():
    return np.random.rand(1, 3, 10).astype(np.float32), 2, 3, 1, 0, 0, 1, True

def generate_nn_conv_transpose2d_input():
    return np.random.rand(1, 3, 10, 10).astype(np.float32), 2, (3, 3), 1, 0, (0, 0), 1, True

def generate_nn_conv_transpose3d_input():
    return np.random.rand(1, 3, 10, 10, 10).astype(np.float32), 2, (3, 3, 3), 1, 0, (0, 0, 0), 1, True

def generate_nn_cosine_embedding_loss_input():
    return np.random.rand(10, 3).astype(np.float32), np.random.rand(10, 3).astype(np.float32), 0.0, 'mean'

def generate_nn_cosine_similarity_input():
    return np.random.rand(10, 3).astype(np.float32), np.random.rand(10, 3).astype(np.float32), 1, 1e-08

def generate_nn_cross_entropy_loss_input():
    return np.random.rand(10, 3).astype(np.float32), np.random.randint(0, 3, size=(10,)).astype(np.int32)

def generate_nn_ctc_loss_input():
    return np.random.rand(10, 3, 20).astype(np.float32), np.random.randint(0, 3, size=(10, 20)).astype(np.int32), np.random.randint(1, 20, size=(10,)).astype(np.int32), np.random.randint(1, 20, size=(10,)).astype(np.int32)
def generate_nn_dropout_input():
    return np.random.rand(1, 10).astype(np.float32), 0.5

def generate_nn_embedding_bag_input():
    return np.random.randint(0, 10, size=(5, 3)).astype(np.int64), np.random.rand(10, 3).astype(np.float32), 1, 2.0, False, 'mean', False, None, None

def generate_nn_embedding_input():
    num_embeddings = 10
    embedding_dim = 5
    return np.random.randint(0, num_embeddings, size=(2, 3)), num_embeddings, embedding_dim

def generate_nn_feature_alpha_dropout_input():
    return np.random.rand(1, 10).astype(np.float32), 0.5

def generate_nn_flatten_input():
    return np.random.rand(1, 2, 3, 4).astype(np.float32)

def generate_nn_fold_input():
    output_size = (2, 3, 4, 5)
    kernel_size = (3, 3)
    return np.random.rand(1, 9, 8, 8).astype(np.float32), output_size, kernel_size

def generate_nn_fractional_max_pool2d_input():
    kernel_size = (2, 2)
    output_size = (4, 4)
    return np.random.rand(1, 8, 8).astype(np.float32), kernel_size, output_size

def generate_nn_fractional_max_pool3d_input():
    kernel_size = (2, 2, 2)
    output_size = (4, 4, 4)
    return np.random.rand(1, 8, 8, 8).astype(np.float32), kernel_size, output_size

def generate_nn_adaptive_avg_pool1d_input():
    output_size = 5
    return np.random.rand(1, 10, 20).astype(np.float32), output_size

def generate_nn_adaptive_avg_pool2d_input():
    output_size = (5, 5)
    return np.random.rand(1, 10, 20, 20).astype(np.float32), output_size

def generate_nn_adaptive_avg_pool3d_input():
    output_size = (5, 5, 5)
    return np.random.rand(1, 10, 20, 20, 20).astype(np.float32), output_size

def generate_layer_norm_input():
    normalized_shape = (3,)
    return np.random.rand(2, 3).astype(np.float32), normalized_shape

def generate_leaky_relu_input():
    return np.random.rand(2, 3).astype(np.float32),

def generate_linear_input():
    input_data = np.random.rand(2, 5).astype(np.float32)
    weight = np.random.rand(4, 5).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    return input_data, weight, bias

def generate_local_response_norm_input():
    return np.random.rand(2, 3, 10, 10).astype(np.float32), 5

def generate_logsigmoid_input():
    return np.random.rand(2, 3).astype(np.float32),

def generate_log_softmax_input():
    return np.random.rand(2, 5).astype(np.float32), 1

def generate_lp_pool1d_input():
    input_data = np.random.rand(1, 3, 10).astype(np.float32)
    kernel_size = 3
    return input_data, kernel_size

def generate_lp_pool2d_input():
    input_data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    kernel_size = (3, 3)
    return input_data, kernel_size

def generate_margin_ranking_loss_input():
    input1 = np.random.rand(10, 5).astype(np.float32)
    input2 = np.random.rand(10, 5).astype(np.float32)
    target = np.random.randint(0, 2, size=(10,)).astype(np.float32)
    return input1, input2, target

def generate_max_pool1d_input():
    input = np.random.rand(1, 3, 10).astype(np.float32)
    kernel_size = 2
    stride = 2
    padding = 0
    return input, kernel_size, stride, padding

def generate_max_pool2d_input():
    input = np.random.rand(1, 3, 10, 10).astype(np.float32)
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    return input, kernel_size, stride, padding

def generate_max_pool3d_input():
    input = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    kernel_size = (2, 2, 2)
    stride = 2
    padding = 0
    return input, kernel_size, stride, padding

def generate_max_unpool1d_input():
    input = np.random.rand(1, 3, 10).astype(np.float32)
    indices = np.random.randint(0, 10, size=(1, 3, 10)).astype(np.int64)
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = (1, 3, 10)
    return input, indices, kernel_size, stride, padding, output_size

def generate_max_unpool2d_input():
    input = np.random.rand(1, 3, 10, 10).astype(np.float32)
    indices = np.random.randint(0, 10, size=(1, 3, 10, 10)).astype(np.int64)
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    output_size = (1, 3, 10, 10)
    return input, indices, kernel_size, stride, padding, output_size

def generate_max_unpool3d_input():
    input = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    indices = np.random.randint(0, 10, size=(1, 3, 10, 10, 10)).astype(np.int64)
    kernel_size = (2, 2, 2)
    stride = 2
    padding = 0
    output_size = (1, 3, 10, 10, 10)
    return input, indices, kernel_size, stride, padding, output_size

def generate_mish_input():
    return np.random.rand(1, 3, 10, 10).astype(np.float32)

def generate_mse_loss_input():
    input = np.random.rand(10, 5).astype(np.float32)
    target = np.random.rand(10, 5).astype(np.float32)
    return input, target

def generate_multilabel_margin_loss_input():
    input = np.random.rand(10, 5).astype(np.float32)
    target = np.random.randint(0, 2, size=(10, 5)).astype(np.float32)
    return input, target

def generate_multilabel_soft_margin_loss_input():
    input = np.random.rand(10, 5).astype(np.float32)
    target = np.random.randint(0, 2, size=(10, 5)).astype(np.float32)
    return input, target