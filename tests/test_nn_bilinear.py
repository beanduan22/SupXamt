from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_bilinear_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_bilinear
from functions.chainer_functions import nn_chainer_bilinear

api_functions = {
    "pytorch_nn_bilinear": nn_torch_bilinear,
    "chainer_nn_bilinear": nn_chainer_bilinear,
}

@given(input_data=generate_nn_bilinear_input())
@settings(max_examples=10, deadline=None)
def test_nn_bilinear_functions(input_data):
    run_test("test_nn_bilinear", input_data, api_functions)

if __name__ == "__main__":
    test_nn_bilinear_functions()
    finalize_results("test_nn_bilinear")
