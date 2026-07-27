import time
import torch
from datetime import datetime
import numpy as np
import tensorflow as tf
import jax.numpy as jnp
import logging
from utilities.logger import log_execution_time, log_results, log_execution_details, save_fail_log, save_total_log
from utilities.counters import Counter
from utilities.summary import summarize_results
from outputs.output_strategy import compare_results

logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

counters = Counter()
results_log = []
exec_times = {}
fail_log = []
total_log = []

def convert_input(input_data, api_name):
    logging.debug(f"Original input for {api_name}: {input_data}")
    if api_name.startswith("pytorch"):
        return input_data if isinstance(input_data, torch.Tensor) else torch.tensor(input_data, dtype=torch.float32)
    elif api_name.startswith("tensorflow"):
        return input_data if isinstance(input_data, tf.Tensor) else tf.convert_to_tensor(input_data, dtype=tf.float32)
    elif api_name.startswith("chainer"):
        return input_data.astype(np.float32)
    elif api_name.startswith("jax"):
        return input_data if isinstance(input_data, jnp.ndarray) else jnp.array(input_data, dtype=jnp.float32)
    elif api_name.startswith("keras"):
        return input_data if isinstance(input_data, np.ndarray) else np.array(input_data, dtype=np.float32)
    return input_data

def run_test(file_name, input_data, api_functions):
    torch_x, np_x, tf_x, jax_x = input_data

    results = {}
    converted_inputs = {
        "pytorch_amin": torch_x,
        "tensorflow_amin": tf_x,
        "chainer_amin": np_x,
        "jax_amin": jax_x,
    }

    for api_name, api_func in api_functions.items():
        try:
            api_input = converted_inputs[api_name]
            logging.debug(f"Converted input for {api_name}: {api_input}")
            api_start_time = time.time()
            result = api_func(api_input)
            exec_time = time.time() - api_start_time
            log_execution_time(file_name, api_name, exec_time)
            if api_name not in exec_times:
                exec_times[api_name] = []
            exec_times[api_name].append(exec_time)
            results[api_name] = {
                "result": result,
                "execution_time": exec_time
            }
        except Exception as e:
            results[api_name] = {
                "result": str(e),
                "execution_time": None
            }

    log_results(file_name, input_data, results)
    
    pytorch_result = results.get("pytorch_amin", {}).get("result", None)
    if pytorch_result is None:
        return

    test_passed = True
    atol = 1e-2
    rtol = 1e-2
    for api_name, data in results.items():
        if "pytorch" not in api_name and data["execution_time"] is not None:
            if not compare_results(pytorch_result, data["result"], atol=atol, rtol=rtol):
                test_passed = False
                fail_log.append({
                    "input": input_data,
                    "results": {k: v["result"] for k, v in results.items()}
                })
    
    total_log.append({
        "input": input_data,
        "results": {k: v["result"] for k, v in results.items()}
    })
    
    counters.increment_correct() if test_passed else counters.increment_incorrect()

    results_log.append({
        "input": input_data,
        "results": {k: {"result": v["result"], "execution_time": v["execution_time"]} for k, v in results.items()},
        "passed": test_passed
    })

def finalize_results(file_name):
    avg_exec_times = {api_name: (sum(times) / len(times)) if len(times) > 0 else 0 for api_name, times in exec_times.items()}
    summary = summarize_results(results_log)
    summary["average_execution_times"] = avg_exec_times
    log_execution_details(file_name, results_log, avg_exec_times, summary)
    save_fail_log(file_name, fail_log)
    save_total_log(file_name, total_log)
    save_counters_with_metadata(counters, "outputs/counters_log.txt", file_name)

def save_counters_with_metadata(counter, file_path, file_name):
    with open(file_path, 'a') as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Correct: {counter.correct}\n")
        f.write(f"Incorrect: {counter.incorrect}\n\n")
