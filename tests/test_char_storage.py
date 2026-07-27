from hypothesis import given, settings
from inputs.input_generator import generate_char_storage_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_char_storage

api_functions = {
}

@given(input_data=generate_char_storage_inputs())
@settings(max_examples=100, deadline=None)
def test_char_storage_functions(input_data):
    run_test("test_char_storage", input_data, api_functions)

if __name__ == "__main__":
    test_char_storage_functions()
    finalize_results("test_char_storage")
