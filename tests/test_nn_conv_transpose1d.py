from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_conv_transpose1d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_conv_transpose1d
from functions.tf_functions import nn_tf_conv_transpose1d

api_functions = {
    "pytorch_nn_conv_transpose1d": nn_torch_conv_transpose1d,
    "tensorflow_nn_conv_transpose1d": nn_tf_conv_transpose1d
}

@given(input_data=generate_nn_conv_transpose1d_input())
@settings(max_examples=10, deadline=None)
def test_nn_conv_transpose1d_functions(input_data):
    run_test("test_nn_conv_transpose1d", input_data, api_functions)

if __name__ == "__main__":
    test_nn_conv_transpose1d_functions()
    finalize_results("test_nn_conv_transpose1d")
