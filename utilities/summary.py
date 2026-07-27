import json
import os
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

def save_summary(summary_data, file_path):
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'w') as file:
        json.dump(summary_data, file, indent=4, default=serialize)

def load_summary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def summarize_results(results):
    summary = {
        "total_tests": len(results),
        "passed_tests": sum(1 for result in results if result["passed"]),
        "failed_tests": sum(1 for result in results if not result["passed"]),
        "details": results
    }
    return summary
