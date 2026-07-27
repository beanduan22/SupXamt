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
        if isinstance(input_data, torch.Tensor):
            return input_data.clone().detach().type(torch.float32)
        return torch.tensor(input_data, dtype=torch.float32)
    elif api_name.startswith("tensorflow"):
        if isinstance(input_data, torch.Tensor):
            return tf.convert_to_tensor(input_data.numpy(), dtype=tf.float32)
        if isinstance(input_data, np.ndarray):
            return tf.convert_to_tensor(input_data, dtype=tf.float32)
    elif api_name.startswith("chainer"):
        if isinstance(input_data, torch.Tensor):
            return input_data.numpy().astype(np.float32)
        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)
    elif api_name.startswith("jax"):
        if isinstance(input_data, torch.Tensor):
            return jnp.array(input_data.numpy(), dtype=jnp.float32)
        if isinstance(input_data, np.ndarray):
            return jnp.array(input_data, dtype=jnp.float32)
    elif api_name.startswith("keras"):
        if isinstance(input_data, torch.Tensor):
            return tf.convert_to_tensor(input_data.numpy(), dtype=tf.float32)
        if isinstance(input_data, np.ndarray):
            return tf.convert_to_tensor(input_data, dtype=tf.float32)
    return input_data

def run_test(file_name, input_data, api_functions):
    if isinstance(input_data, (list, tuple)):
        inputs = input_data
    else:
        inputs = (input_data,)

    results = {}
    for api_name, api_func in api_functions.items():
        try:
            api_input = [convert_input(data, api_name) for data in inputs]
            logging.debug(f"Converted input for {api_name}: {api_input}")
            api_start_time = time.time()
            result = api_func(*api_input)
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
    
    pytorch_result = None
    for api_name, data in results.items():
        if "pytorch" in api_name:
            pytorch_result = data["result"]
            break
    
    if pytorch_result is None:
        return

    test_passed = True
    for api_name, data in results.items():
        if "pytorch" not in api_name and data["execution_time"] is not None:
            if not compare_results(pytorch_result, data["result"], atol=1e-100, rtol=1e-100):
                test_passed = False
                fail_log.append({
                    "input": [tensor.tolist() if hasattr(tensor, 'tolist') else tensor for tensor in inputs],
                    "results": {k: (v["result"].tolist() if hasattr(v["result"], 'tolist') else v["result"]) for k, v in results.items()}
                })
    
    total_log.append({
        "input": [tensor.tolist() if hasattr(tensor, 'tolist') else tensor for tensor in inputs],
        "results": {k: (v["result"].tolist() if hasattr(v["result"], 'tolist') else v["result"]) for k, v in results.items()}
    })
    
    counters.increment_correct() if test_passed else counters.increment_incorrect()

    results_log.append({
        "input": [tensor.tolist() if hasattr(tensor, 'tolist') else tensor for tensor in inputs],
        "results": {k: {"result": (v["result"].tolist() if hasattr(v["result"], 'tolist') else v["result"]),
                        "execution_time": v["execution_time"]} for k, v in results.items()},
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
