from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_alpha_dropout_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_alpha_dropout
from functions.tf_functions import nn_tf_alpha_dropout
from functions.keras_functions import nn_keras_alpha_dropout
from functions.chainer_functions import nn_chainer_dropout
from functions.jax_functions import nn_jax_dropout

api_functions = {
    "pytorch_nn_alpha_dropout": nn_torch_alpha_dropout,
    "tensorflow_nn_alpha_dropout": nn_tf_alpha_dropout,
    "keras_nn_alpha_dropout": nn_keras_alpha_dropout,
    "chainer_nn_dropout": nn_chainer_dropout,
    "jax_nn_dropout": nn_jax_dropout,
}

@given(input_data=generate_nn_alpha_dropout_input())
@settings(max_examples=10, deadline=None)
def test_nn_alpha_dropout_functions(input_data):
    run_test("test_nn_alpha_dropout", input_data, api_functions)

if __name__ == "__main__":
    test_nn_alpha_dropout_functions()
    finalize_results("test_nn_alpha_dropout")
