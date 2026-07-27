from hypothesis import given, settings
from inputs.input_generator import generate_atan2_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atan2
from functions.tf_functions import tf_atan2
from functions.jax_functions import jax_atan2
from functions.chainer_functions import chainer_atan2
from functions.keras_functions import keras_atan2

api_functions = {
    "pytorch_atan2": torch_atan2,
    "tensorflow_atan2": tf_atan2,
    "jax_atan2": jax_atan2,
}

@given(input_data=generate_atan2_inputs())
@settings(max_examples=100, deadline=None)
def test_atan2_functions(input_data):
    run_test("test_atan2", input_data, api_functions)

if __name__ == "__main__":
    test_atan2_functions()
    finalize_results("test_atan2")
