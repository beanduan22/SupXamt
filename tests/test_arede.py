import torch
import tensorflow as tf
import chainer

from hypothesis import given, settings
import hypothesis.strategies as st

import jax

def torch_are_deterministic_algorithms_enabled():
    return torch.are_deterministic_algorithms_enabled()

def tf_executing_eagerly():
    return tf.executing_eagerly()

def chainer_is_debug():
    return chainer.is_debug()

def jax_config_omnistaging_enabled():
    return jax.config.omnistaging

@given(dummy_input=st.integers())
@settings(deadline=None)
def test_framework_functions(dummy_input):
    torch_output = torch_are_deterministic_algorithms_enabled()
    tf_output = tf_executing_eagerly()
    chainer_output = chainer_is_debug()
    jax_output = jax_config_omnistaging_enabled()

    if not (torch_output == tf_output == chainer_output == jax_output):
        print("Inconsistent result found!")
        print("PyTorch Output:", torch_output)
        print("TensorFlow Output:", tf_output)
        print("Chainer Output:", chainer_output)
        print("JAX Output:", jax_output)

if __name__ == "__main__":
    test_framework_functions()
