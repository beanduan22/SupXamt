import os
import json
from datetime import datetime
import numpy as np

def serialize(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if hasattr(obj, 'numpy'):
        return obj.numpy().tolist()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'data'):
        return obj.data.tolist()
    return str(obj)

def ensure_directory_exists(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def log_execution_details(file_name, results_log, avg_exec_times, summary):
    log_path = f"outputs/{file_name}_execution_log.json"
    ensure_directory_exists(log_path)
    
    log_data = {
        "Execution File": file_name,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": summary,
        "execution_details": results_log,
        "average_execution_times": avg_exec_times,
        "total_tests": summary['total_tests'],
        "passed_tests": summary['passed_tests'],
        "failed_tests": summary['failed_tests']
    }

    with open(log_path, 'a') as f:
        json.dump(log_data, f, indent=4, default=serialize)

    log_summary = {
        "Execution File": file_name,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Summary Report": {
            "Total Test Cases": summary['total_tests'],
            "Passed Test Cases": summary['passed_tests'],
            "Failed Test Cases": summary['failed_tests'],
            "Average Execution Time": avg_exec_times
        }
    }

    summary_log_path = f"outputs/{file_name}_summary_log.json"
    ensure_directory_exists(summary_log_path)
    with open(summary_log_path, 'a') as f:
        json.dump(log_summary, f, indent=4, default=serialize)

def save_fail_log(file_name, fail_log):
    fail_log_path = f"outputs/task/{file_name}_fail.json"
    ensure_directory_exists(fail_log_path)
    log_data = {
        "Execution File": file_name,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Failed Cases": fail_log
    }
    with open(fail_log_path, 'a') as f:
        json.dump(log_data, f, indent=4, default=serialize)

def save_total_log(file_name, total_log):
    total_log_path = f"outputs/task/{file_name}_total.json"
    ensure_directory_exists(total_log_path)
    log_data = {
        "Execution File": file_name,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "All Cases": total_log
    }
    with open(total_log_path, 'a') as f:
        json.dump(log_data, f, indent=4, default=serialize)

def log_execution_time(file_name, api_name, exec_time):
    log_path = "outputs/execution_time_log.json"
    ensure_directory_exists(log_path)
    log_data = {
        "Execution File": file_name,
        "API Name": api_name,
        "Execution Time": exec_time,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_data, indent=4) + "\n")

def log_results(file_name, input_data, results):
    log_path = "outputs/results_log.json"
    ensure_directory_exists(log_path)
    log_data = {
        "Execution File": file_name,
        "Input Data": input_data,
        "Results": results,
        "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_data, indent=4, default=serialize) + "\n")
