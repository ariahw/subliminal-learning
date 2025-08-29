import os 
import orjson
import json
from pydantic import BaseModel


RESULT_FILEPATH = "results"

if not os.path.exists(RESULT_FILEPATH):
    os.makedirs(RESULT_FILEPATH)


'''UTILS FUNCTIONS'''

def results_path(output_fpath: str):
    return f"{RESULT_FILEPATH}/{output_fpath}"


def verify_path(path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def jsonify(dataset):
    if isinstance(dataset[0], BaseModel):
        return "\n".join([x.model_dump_json() for x in dataset])
    else:
        return "\n".join([str(json.dumps(x)) for x in dataset])


def save_dataset_jsonl(dataset: list[BaseModel] | list[dict], filename: str, overwrite: bool = True):
    '''Append to existing dataset or create a new one if it doesn't exist'''
    
    verify_path(filename)
    
    if overwrite or (not os.path.exists(filename)):
        with open(filename, "w") as f:
            f.write(jsonify(dataset))
    else:
        with open(filename, "a") as f:
            f.write("\n") # Go to next line
            f.write(jsonify(dataset))


def read_jsonl_all(filename: str) -> list[dict]:
    '''For debugging use - defeats purpose of format in terms of sizing'''

    with open(filename, "r") as f:
        lines = f.readlines()
    
    return [orjson.loads(line) for line in lines if line.strip()]


def count_lines(filename: str) -> int:
    '''Count the number of lines in a file'''

    if not os.path.exists(filename):
        return 0
    else:
        with open(filename, "r") as f:
            return sum(1 for _ in f)


