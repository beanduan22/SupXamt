from hypothesis import given, settings
from inputs.input_generator import addbmm_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_addbmm
from functions.tf_functions import tf_addbmm
from functions.keras_functions import keras_addbmm
from functions.jax_functions import jax_addbmm

api_functions = {
    "tensorflow_addbmm": tf_addbmm,
    "keras_addbmm": keras_addbmm,
    "jax_addbmm": jax_addbmm,
}

@given(input_data=addbmm_input_strategy())
@settings(max_examples=100, deadline=None)
def test_addbmm_functions(input_data):
    run_test("test_addbmm", input_data, api_functions)

if __name__ == "__main__":
    test_addbmm_functions()
    finalize_results("test_addbmm")
