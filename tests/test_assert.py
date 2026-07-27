import torch
import tensorflow as tf
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st

def torch_arcsin(x):
    return torch.asin(x)

def torch_arctan(x):
    return torch.atan(x)

def torch_arctanh(x):
    return torch.atanh(x)

def torch_any(input):
    return torch.any(input)

def torch_argmax(input):
    return torch.argmax(input)

def torch_argmin(input):
    return torch.argmin(input)

def torch_argsort(values):
    return torch.argsort(values)

def torch_asinh(x):
    return torch.asinh(x)

def torch_assert(condition, message):
    assert condition.item(), message

def tf_arcsin(x):
    return tf.asin(x)

def tf_arctan(x):
    return tf.atan(x)

def tf_arctanh(x):
    return tf.atanh(x)

def tf_reduce_any(input):
    return tf.reduce_any(input)

def tf_argmax(input):
    return tf.argmax(input)

def tf_argmin(input):
    return tf.argmin(input)

def tf_argsort(values):
    return tf.argsort(values)

def tf_asinh(x):
    return tf.asinh(x)

def tf_assert(condition, message):
    assert tf.reduce_all(condition), message

def jax_arcsin(x):
    return jnp.arcsin(x)

def jax_arctan(x):
    return jnp.arctan(x)

def jax_arctanh(x):
    return jnp.arctanh(x)

def jax_any(input):
    return jnp.any(input)

def jax_argmax(input):
    return jnp.argmax(input)

def jax_argmin(input):
    return jnp.argmin(input)

def jax_argsort(values):
    return jnp.argsort(values)

def jax_asinh(x):
    return jnp.arcsinh(x)

def jax_assert(condition, message):
    assert jnp.all(condition), message

@given(x=st.floats(min_value=-1.0, max_value=1.0))
@settings(deadline=None)
def test_arcsin_functions(x):
    torch_output = torch_arcsin(torch.tensor(x))
    tf_output = tf_arcsin(x)
    jax_output = jax_arcsin(x)
    
    if not torch_output == tf_output == jax_output:
        print("Inconsistent result found!")
        print("Input:", x)
        print("PyTorch Output:", torch_output)
        print("TensorFlow Output:", tf_output)
        print("JAX Output:", jax_output)
        print()


if __name__ == "__main__":
    test_arcsin_functions()
