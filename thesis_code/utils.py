import pandas
import numpy as np
import os
import csv

def save_results(path, engine, model, time):
    "CSV with fields (engine, model, time)? appends to csv?"
    save_path = os.path.join(path, "results.csv")
    fields = [engine, model, time]
    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)