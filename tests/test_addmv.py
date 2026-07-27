from hypothesis import given, settings
from inputs.input_generator import generate_addmv_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_addmv
from functions.tf_functions import tf_addmv
from functions.chainer_functions import chainer_addmv
from functions.keras_functions import keras_addmv
from functions.jax_functions import jax_addmv

api_functions = {
    "pytorch_addmv": torch_addmv,
    "tensorflow_addmv": tf_addmv,
    "chainer_addmv": chainer_addmv,
    "jax_addmv": jax_addmv,
}

@given(input_data=generate_addmv_inputs())
@settings(max_examples=100, deadline=None)
def test_addmv_functions(input_data):
    run_test("test_addmv", input_data, api_functions)

if __name__ == "__main__":
    test_addmv_functions()
    finalize_results("test_addmv")
