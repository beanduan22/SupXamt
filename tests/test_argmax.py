from hypothesis import given, settings
from inputs.input_generator import generate_argmax_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_argmax
from functions.tf_functions import tf_argmax
from functions.chainer_functions import chainer_argmax
from functions.keras_functions import keras_argmax
from functions.jax_functions import jax_argmax

api_functions = {
    "pytorch_argmax": torch_argmax,
    "tensorflow_argmax": tf_argmax,
    "keras_argmax": keras_argmax,
    "jax_argmax": jax_argmax,
}

@given(input_data=generate_argmax_inputs())
@settings(max_examples=100, deadline=None)
def test_argmax_functions(input_data):
    run_test("test_argmax", input_data, api_functions)

if __name__ == "__main__":
    test_argmax_functions()
    finalize_results("test_argmax")
