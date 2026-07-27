import subprocess
import sys
import os

test_files = [
    "tests/test_abs.py",
    "tests/test_absolute.py",
    "tests/test_acos.py",
    "tests/test_acosh.py",
    "tests/test_add.py",
    "tests/test_addbmm.py",
    "tests/test_addcmul.py",
    "tests/test_addmm.py",
    "tests/test_all.py",
]

def run_test(file):
    result = subprocess.run([sys.executable, file], capture_output=True, text=True)
    return result

def run_all_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    for test_file in test_files:
        print(f"Running {test_file}...")
        result = run_test(test_file)
        print(f"Finished {test_file}")
        print("Output:")
        print(result.stdout)
        print("Errors:")
        print(result.stderr)
        print("=" * 40)

if __name__ == "__main__":
    run_all_tests()
