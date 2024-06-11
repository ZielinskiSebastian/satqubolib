from satqubolib.formula import CNF
import os
from pathlib import Path


def load_formula(method, num_clauses, formula_number):
    file_path = Path(__file__)
    path = os.path.join(file_path.parent, method, str(num_clauses), str(formula_number) + ".cnf")
    return CNF.from_file(path)
