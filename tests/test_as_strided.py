from hypothesis import given, settings
from inputs.input_generator import generate_as_strided_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_as_strided
from functions.tf_functions import tf_strided_slice

api_functions = {
}

@given(input_data=generate_as_strided_inputs())
@settings(max_examples=100, deadline=None)
def test_as_strided_functions(input_data):
    run_test("test_as_strided", input_data, api_functions)

if __name__ == "__main__":
    test_as_strided_functions()
    finalize_results("test_as_strided")