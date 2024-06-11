"""
This module documents and explains the satqubolib.formula module.

IMPORTANT NOTE: The current version of satqubolib only supports 3-SAT formulas containing exactly 3 literals per clause.
"""

from satqubolib.formula import CNF
from dataset import dataset_loader
"""
Example 1 - Creating a CNF object
"""

def example_1():
    # A 3-SAT formula is a defined by its clauses.
    # Suppose we are given the following 3-SAT formula: (x1 OR x2 OR x3) AND (x1 OR NOT x2 OR NOT x4)
    # We can represent this formula as a CNF object as follows:

    cnf = CNF([[1, 2, 3], [1, -2, -4]])
    print(cnf)

    # Each clause is represented as a list of literals (integers) -  literals are always non-zero integers
    # IMPORTANT: in the current version of satqubolib, each clause must contain exactly 3 literals.

    # If we want to load a CNF formula from a file in DIMACS format, we can use the CNF.from_file() method.
    # In this example, we will load the formula specified in the example_cnf1.dimacs file
    cnf2 = CNF.from_file("example_cnf1.dimacs")
    print(cnf2)


    # Loading a formula from satqubolib's dataset
    method = "balanced"
    num_clauses = 1000
    formula_number = 1

    cnf3 = dataset_loader.load_formula(method, num_clauses, formula_number)
    print(cnf3)

    ###################
    # IMPORTANT: in the current version of satqubolib, each clause must contain exactly 3 literals.
    ###################

"""
Example 2 - Adding and Removing Clauses
"""


def example_2():
    cnf = CNF([[1, 2, 3], [1, -2, -4]])
    # If we already have a CNF object, we can add and remove clauses from it:
    cnf.add_clause([5, 6, -7])
    print(cnf)
    cnf.remove_clause([5, 6, -7])
    print(cnf)

"""
Example 3 - Checking if an assignment is a solution
"""

def example_3():
    # Suppose, we have an assignment of values to the variables of the formula.
    # An assignment is a dictionary where the keys are the variable numbers and the values are the assigned values (0 or 1).
    # Suppose we are given the fomrula: (x1 OR x2 OR x3) AND (x1 OR NOT x2 OR NOT x4)
    # and the assignment: {1: 1, 2: 0, 3: 1, 4: 0}
    # We can check if this assignment is a solution to the formula as follows:

    cnf = CNF([[1, 2, 3], [1, -2, -4]])
    assignment = {1: 1, 2: 0, 3: 1, 4: 0}
    is_satisfied = cnf.is_satisfied(assignment)
    print(is_satisfied)

"""
Example 4 - Counting the number of satisfied clauses
"""

def example_4():
    # There are two cases, where we might be interested in knowing how many clauses a given assignment satisfies:
    # 1. We are solving MAX-SAT
    # 2. We are solving SAT and a solver did not return a satisfying solution yet, and we are interested in the best
    #    solution so far.
    # In both cases, we can count the number of satisfied clauses as follows:

    cnf = CNF([[1, 2, 3], [1, -2, -4]])
    assignment = {1: 1, 2: 0, 3: 1, 4: 0}
    num_satisfied = cnf.count_satisfied_clauses(assignment)
    print(num_satisfied)  # returns 2, because the assignment satisfies 2 out of 2 clauses

"""
Example 5 - Accessing metadata
"""

def example_5():
    # Each formula of satqubolib's dataset contains the following metadata:
    # 1. solution - the solution to the formula
    # 2. solve_time - the time it took to solve the formula
    # 3. solver - the solver used to solve the formula
    # 4. cpu - the CPU used to solve the formula
    # NOTE: If a formula is loaded from a file and the metadata is present in the file, the metadata will be automatically
    #       added to the CNF object.

    # If you open example_cnf2.dimacs, you will see the following header:

    # c Solution: 1 2 3
    # c Time: 1573.86 seconds
    # c Solver: Kissat 3.1.0
    # c CPU: AMD_Ryzen_Threadripper_PRO_5965WX_4.5_GHZ

    # Suppose we have loaded a formula from a file and we want to access the metadata, we can do so as follows:

    cnf3 = CNF.from_file("example_cnf2.dimacs")
    solution = cnf3.solution
    solve_time = cnf3.solve_time
    solver = cnf3.solver
    cpu = cnf3.cpu
    print("Solution: ", solution, "\n", "Solve Time: ", solve_time, "\n", "Solver: ", solver, "\n", "CPU: ", cpu)

    # Similarly if a new CNF object without metadata is created, we can add the metadata, by assigning values to the
    # respective variables

    cnf4 = CNF([[1, 2, 3], [1, -2, -4]])
    cnf4.solution = [1, 0, 1, 0]
    cnf4.solve_time = 0.1
    cnf4.solver = "MiniSat"
    cnf4.cpu = "Intel Core i11"
    print("Solution: ", cnf4.solution, "\n", "Solve Time: ", cnf4.solve_time, "\n", "Solver: ", cnf4.solver, "\n", "CPU: ", cnf4.cpu)

"""
Example 6 - Saving a CNF formula to a file
"""


def example_6():
    # Suppose we have created a CNF object (as shown in Example 1) and we want to save it to a file.
    # We can use the to_file() method as follows:
    cnf = CNF([[1, 2, 3], [1, -2, -4]])
    cnf.to_file("example_cnf3.dimacs")

    # If metadata is known, we can add it to the CNF object (as shown in Example 5). When we save the CNF object to a file,
    # the metadata will be saved as the header of the file automatically.

    cnf4 = CNF([[1, 2, 3], [1, -2, -4]])
    cnf4.solution = [1, 0, 1, 0]
    cnf4.solve_time = 0.1
    cnf4.solver = "MiniSat"
    cnf4.cpu = "Intel Core i11"

    cnf4.to_file("example_cnf4.dimacs")

    # if you now open the file example_cnf4.dimacs, you will see the following header:
    # c Solution: 1 0 1 0
    # c Time: 0.1 seconds
    # c Solver: MiniSat
    # c CPU: Intel Core i11


if __name__ == "__main__":
    example_1()
    # example_2()
    # example_3()
    # example_4()
    # example_5()
    # example_6()
