import json
import os

class Counter:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def increment_correct(self):
        self.correct += 1

    def increment_incorrect(self):
        self.incorrect += 1

    def get_summary(self):
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
        }

def save_counters(counters, file_path):
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'w') as file:
        json.dump(counters.get_summary(), file)
