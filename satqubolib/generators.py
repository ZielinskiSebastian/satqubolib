import random
from satqubolib.formula import CNF
import numpy as np


class RandomSAT:
    """
    This class generates a random SAT formula with the given number of variables and clauses.

    Parameters:
    num_variables (int): The number of variables in the formula.
    num_clauses (int): The number of clauses in the formula.
    vars_per_clause (int): The number of variables per clause. Default is 3.

    NOTE: SATQUBOLIB currently supports only 3-SAT formulas containing exactly 3 literals per clause.
    """
    def __init__(self, num_vars, num_clauses, vars_per_clause=3):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.vars_per_clause = vars_per_clause

    def generate(self):
        formula = []
        for clause_index in range(self.num_clauses):
            clause_vars = np.random.choice(range(1, self.num_vars + 1), size=3, replace=False)
            signs = np.random.choice([-1, +1], size=3, replace=True)
            formula.append([x.item() for x in clause_vars * signs])
        return CNF(formula)


class BalancedNoTriangleBase:
    """
    This class is the base class for the BalancedSAT and NoTriangleSAT classes. It contains the common methods for both
    (BalancedSAT and NoTriangleSAT) classes.

    NOTE: SATQUBOLIB currently supports only 3-SAT formulas containing exactly 3 literals per clause.
    """

    def __init__(self, num_vars, num_clauses, vars_per_clause,):
        self.num_vars = num_vars
        self.vars_per_claus = vars_per_clause
        self.num_clauses = num_clauses

    def find_least_frequent_variables(self, variable_occurrences):
        min_occurrence = min(variable_occurrences.values())
        return [variable for variable, num_occurrences in variable_occurrences.items() if
                num_occurrences == min_occurrence]

    def calculate_new_pairs(self, clause, variable):
        new_pairs = []
        for other_variable in clause:
            if variable != other_variable:
                new_pairs.append((variable, other_variable))

        return new_pairs

    def find_least_repeated_pairs(self, pair_list, current_clause, variables):

        variable_pair_repetition = {variable: 0 for variable in variables}
        for variable in variables:
            new_pairs = self.calculate_new_pairs(current_clause, variable)

            for pair in new_pairs:
                if pair[0] in pair_list[pair[1]]:
                    variable_pair_repetition[variable] += 1

        return self.find_least_frequent_variables(variable_pair_repetition)

    def add_variable_to_clause(self, formula, variable, clause_index, variable_index, variable_occurrences,
                               variable_occurrences_clause, pair_list):
        formula[clause_index].append(variable)
        variable_occurrences[variable] += 1

        # prevent the algorithm to choose the same variable multiple times in a clause
        del variable_occurrences_clause[variable]

        # The addition of the first variable to a clause cannot introduce new pairs
        # The second variable can introduce one new pair, the third variable can introduce two new pairs
        if variable_index > 0:
            new_pairs = self.calculate_new_pairs(formula[clause_index], variable)

            for pair in new_pairs:
                if pair[0] not in pair_list[pair[1]]:
                    pair_list[pair[0]].add(pair[1])
                    pair_list[pair[1]].add(pair[0])

    def apply_signs(self, formula):
        variable_signs = {variable: None for variable in range(self.num_vars)}

        # Calculate signs of literals within the clauses according to the paper
        for clause_index in range(self.num_clauses):
            for variable_index in range(self.vars_per_claus):
                variable = formula[clause_index][variable_index]
                if variable_signs[variable - 1] is None:
                    var_sign = random.randint(0, 1)
                    formula[clause_index][variable_index] = variable * (-1) ** var_sign
                    variable_signs[variable - 1] = var_sign ^ 1
                else:
                    formula[clause_index][variable_index] = variable * (-1) ** variable_signs[variable - 1]
                    variable_signs[variable - 1] = variable_signs[variable - 1] ^ 1
        return CNF(formula)


class BalancedSAT(BalancedNoTriangleBase):
    """
    This class generates 3-SAT formulas according to the Balanced SAT method:
    Paper Name: Balanced Random SAT Benchmarks
    Paper Authors: Ivor Spence
    Paper Link: https://helda.helsinki.fi/bitstreams/93e6d772-8fb3-4502-823c-97adfb674617/download#page=53

    NOTE: SATQUBOLIB currently supports only 3-SAT formulas containing exactly 3 literals per clause.
    """

    def __init__(self, num_vars, num_clauses, vars_per_clause=3):
        super().__init__(num_vars=num_vars, num_clauses=num_clauses, vars_per_clause=vars_per_clause)

    def generate(self):
        formula = [[] for _ in range(self.num_clauses)]
        variable_occurrences = {k: 0 for k in range(1, self.num_vars + 1)}

        # For each variable we keep track of all pairs that contain this variable
        # Note: we save every pair twice. For the pair (x,y) the value y will be added to the set of pairs of x and
        # vice versa. Thus pair_list[x] contains y and pair_list[y] contains x. This introduces redundancy from a memory
        # perspective, but it improves the performance of searching for constraint triangles significantly.
        pair_list = {variable: set() for variable in range(1, self.num_vars + 1)}

        for clause_index in range(self.num_clauses):
            variable_occurrences_clause = variable_occurrences.copy()

            for variable_index in range(self.vars_per_claus):

                min_vars = self.find_least_frequent_variables(variable_occurrences_clause)

                # If this is the first variable of the clause, it cannot introduce new pairs / constraint triangles
                # -> choose one of the least frequent variables

                if variable_index == 0:
                    self.add_variable_to_clause(formula, min_vars[random.randint(0, len(min_vars) - 1)], clause_index,
                                                variable_index, variable_occurrences, variable_occurrences_clause,
                                                pair_list)
                    continue

                # Search variables that occurred least often
                if len(min_vars) == 1:
                    self.add_variable_to_clause(formula, min_vars[0], clause_index, variable_index,
                                                variable_occurrences, variable_occurrences_clause, pair_list)
                    continue

                # If this is not the first variable of the clause, we need to check whether it introduces repeated pairs

                min_vars = self.find_least_repeated_pairs(pair_list, formula[clause_index], min_vars)
                self.add_variable_to_clause(formula, min_vars[random.randint(0, len(min_vars) - 1)], clause_index,
                                            variable_index, variable_occurrences, variable_occurrences_clause,
                                            pair_list)

        return self.apply_signs(formula)


class NoTriangleSAT(BalancedNoTriangleBase):
    """
    This class generates 3-SAT formulas according to the No Triangle SAT method:
    Paper Name: Generating Difficult SAT Instances by Preventing Triangles
    Paper Authors: G. Escamocher, B. O'Sullivan, S. Prestwich
    Paper Link: https://arxiv.org/pdf/1903.03592
    """

    def __init__(self, num_vars, num_clauses, vars_per_clause=3):
        super().__init__(num_vars=num_vars, num_clauses=num_clauses, vars_per_clause=vars_per_clause)

    def _find_least_added_constraint_triangles(self, pair_list, current_clause, variables):
        variable_triangle_additions = {variable: 0 for variable in variables}

        for variable in variables:
            new_pairs = self.calculate_new_pairs(current_clause, variable)
            for pair in new_pairs:
                # A pair that is already in the set of pairs, cannot introduce new constraint triangles
                if pair[0] in pair_list[pair[1]]:
                    continue

                variable_triangle_additions[variable] += len(pair_list[pair[0]].intersection(pair_list[pair[1]]))

        return self.find_least_frequent_variables(variable_triangle_additions)

    def generate(self):
        formula = [[] for _ in range(self.num_clauses)]
        variable_occurrences = {k: 0 for k in range(1, self.num_vars + 1)}

        # For each variable we keep track of all pairs that contain this variable
        # Note: we save every pair twice. For the pair (x,y) the value y will be added to the set of pairs of x and
        # vice versa. Thus pair_list[x] contains y and pair_list[y] contains x. This introduces redundancy from a memory
        # perspective, but it improves the performance of searching for constraint triangles significantly.
        pair_list = {variable: set() for variable in range(1, self.num_vars + 1)}

        for clause_index in range(self.num_clauses):
            variable_occurrences_clause = variable_occurrences.copy()

            for variable_index in range(self.vars_per_claus):

                min_vars = self.find_least_frequent_variables(variable_occurrences_clause)

                # If this is the first variable of the clause, it cannot introduce new pairs / constraint triangles
                # -> choose one of the least frequent variables

                if variable_index == 0:
                    self.add_variable_to_clause(formula, min_vars[random.randint(0, len(min_vars) - 1)], clause_index,
                                                variable_index, variable_occurrences, variable_occurrences_clause,
                                                pair_list)
                    continue

                # Search variables that occurred least often
                if len(min_vars) == 1:
                    self.add_variable_to_clause(formula, min_vars[0], clause_index, variable_index,
                                                variable_occurrences, variable_occurrences_clause, pair_list)
                    continue

                # If this is not the first variable of the clause, we need to check whether it introduces repeated pairs
                if len(min_vars) == 1:
                    min_vars = self.find_least_repeated_pairs(pair_list, formula[clause_index], min_vars)
                    self.add_variable_to_clause(formula, min_vars[0], clause_index, variable_index,
                                                variable_occurrences, variable_occurrences_clause, pair_list)
                    continue

                # If some variables introduce the same amount of repeated pairs, we check for constraint triangles

                min_vars = self._find_least_added_constraint_triangles(pair_list, formula[clause_index], min_vars)
                self.add_variable_to_clause(formula, min_vars[random.randint(0, len(min_vars) - 1)], clause_index,
                                            variable_index, variable_occurrences, variable_occurrences_clause,
                                            pair_list)

        return self.apply_signs(formula)
