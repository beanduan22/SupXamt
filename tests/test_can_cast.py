from hypothesis import given, settings
from inputs.input_generator import generate_can_cast_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_can_cast
from functions.tf_functions import tf_can_cast

api_functions = {
}

@given(input_data=generate_can_cast_inputs())
@settings(max_examples=100, deadline=None)
def test_can_cast_functions(input_data):
    run_test("test_can_cast", input_data, api_functions)

if __name__ == "__main__":
    test_can_cast_functions()
    finalize_results("test_can_cast")
