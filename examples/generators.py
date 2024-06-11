"""
This module documents and explains the satqubolib.generators module.


"""

from satqubolib.generators import RandomSAT, BalancedSAT, NoTriangleSAT
from pysat.solvers import Glucose4


"""
Example 1 - Generating Random CNF Formulas
"""


def example_1():
    # RandomSAT generates random CNF formulas with a given number of variables and clauses.

    random_sat_generator = RandomSAT(num_vars=5, num_clauses=10, vars_per_clause=3)
    cnf = random_sat_generator.generate()
    print(cnf)
    # The created formula can also be saved easily to a file via:
    # cnf.to_file("myformula.cnf")

    # cnf is a satqubolib.formula.CNF object. Please refer to the formula module documentation for more information.


"""
Example 2 - Generating CNF formulas with the Balanced SAT method
"""


def example_2():
    balanced_sat_generator = BalancedSAT(num_vars=5, num_clauses=10, vars_per_clause=3)
    cnf = balanced_sat_generator.generate()
    print(cnf)
    # The created formula can also be saved easily to a file via:
    # cnf.to_file("myformula.cnf")

    # cnf is a satqubolib.formula.CNF object. Please refer to the formula module documentation for more information.

"""
Example 3 - Generating CNF formulas with the No Triangle SAT method
"""


def example_3():
    no_triangle_sat_generator = NoTriangleSAT(num_vars=3, num_clauses=10, vars_per_clause=3)
    cnf = no_triangle_sat_generator.generate()
    print(cnf)
    # The created formula can also be saved easily to a file via:
    # cnf.to_file("myformula.cnf")

    # cnf is a satqubolib.formula.CNF object. Please refer to the formula module documentation for more information.

"""
Example 4 - Using PySAT to check hardness and satisfiability of generated formulas
"""


def example_4():
    cnf = RandomSAT(num_vars=5, num_clauses=10, vars_per_clause=3).generate()
    # cnf = BalancedSAT(num_vars=5, num_clauses=10, vars_per_clause=3).generate()
    # cnf = NoTriangleSAT(num_vars=3, num_clauses=10, vars_per_clause=3).generate()

    with Glucose4(bootstrap_with=cnf.clauses, use_timer=True) as solver:
        if solver.solve():
            stats = solver.accum_stats()
            print("SAT")
            print('Solution:', solver.get_model())
            print("TIME: ", solver.time_accum())
            print("Stats: ", stats)
        else:
            print("UNSAT")
            print("TIME: ", solver.time_accum())
            print("Stats: ", solver.accum_stats())


if __name__ == "__main__":
    example_1()
    # example_2()
    # example_3()
    # example_4()