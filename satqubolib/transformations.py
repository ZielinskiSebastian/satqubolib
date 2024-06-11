import numpy as np
from satqubolib.formula import CNF
import multiprocessing as mp
from multiprocessing import Pool
import functools
import pickle
from copy import deepcopy


class _QUBOTransformationBase:

    def __init__(self, cnf: CNF):
        """
        Initialize the QUBO transformation base object.

        Args:
            cnf (CNF): A CNF object (3-SAT formula in conjunctive normal form) for which a QUBO matrix is to be created.

        Attributes:
            cnf (CNF): Stores the CNF object passed during initialization.
            qubo (dict): A dictionary to store the quadratic unconstrained binary optimization (QUBO) matrix.
        """
        self.cnf = cnf
        self.qubo = {}


    def add(self, x, y, value):
        """
        Adds or updates a value in the QUBO matrix at a specific position.


        Args:
            x (int): The row index in the QUBO matrix.
            y (int): The column index in the QUBO matrix.
            value (float): The value to be added to the QUBO matrix at the specified indices.
        """
        x, y = abs(x), abs(y)
        if x > y:
            x, y = y, x
        self.qubo[(x, y)] = self.qubo.get((x, y), 0) + value

    def _is_solution(self, solution_candidate: dict):
        """
        Checks if a given solution candidate satisfies the CNF formula.

        Args:
            solution_candidate (dict): A dictionary representing a candidate solution to the CNF problem found by a
            QUBO solver.

        Returns:
            tuple: A tuple where the first element is a boolean indicating if the solution is valid,
                   and the second element is the count of satisfied clauses.
        """

        satisfied_clauses = self.cnf.count_satisfied_clauses(solution_candidate)
        if satisfied_clauses < len(self.cnf.clauses):
            return False, satisfied_clauses
        else:
            return True, satisfied_clauses

    def _get_satisfied_clauses(self, solution_candidate: dict):
        return self.cnf.count_satisfied_clauses(solution_candidate)

    def _print_qubo(self, variable_labels: dict = None):
        qubo_vars = sorted(list(set(sum(self.qubo.keys(), ()))))
        # Print the header row
        header_row = ["QUBO"] + [variable_labels[k] for k in qubo_vars]
        print("{:<8}".format(header_row[0]), " ".join("{:>7}".format(var) for var in header_row[1:]))

        # Print the QUBO matrix rows
        for i in qubo_vars:
            i_idx = qubo_vars.index(i)

            if variable_labels is None:
                print("{:<8}".format(f"Var {i}"), end=" ")
            else:
                print("{:<8}".format(variable_labels[i]), end=" ")

            for j in qubo_vars:
                if  qubo_vars.index(j) >= i_idx:
                    print("{:>7}".format(self.qubo.get((i, j), 0)), end=" ")
                else:
                    print("{:>7}".format(0), end=" ")
            print("")


class ChoiSAT(_QUBOTransformationBase):
    """
    This class provides an implementation of Choi's QUBO transformation for (MAX)-3SAT problems as described in the
    paper: https://arxiv.org/pdf/1004.2226.

    NOTE: This is also the 3SAT method that is described in Lucas' seminal paper "Ising formulations of many NP
    problems": https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full
    """

    def __init__(self, cnf: CNF, intra_clause_weight=3, inter_clause_weight=4):
        super().__init__(cnf)
        self.literals = [literal for clause in self.cnf.clauses for literal in clause]

        self.vertex_weight = -1
        self.intra_clause_weight = intra_clause_weight  # Penalizes multiple set literals in one clause
        self.inter_clause_weight = inter_clause_weight  # Penalizes the occurrence of contradictions

    def create_qubo(self):
        for i in range(len(self.literals)):
            for j in range(len(self.literals)):
                if i > j:
                    continue
                # Main diagonal
                if i == j:
                    self.qubo[(i, j)] = self.vertex_weight
                # Triangle penalties
                elif j - i <= 2 and j//3 == i//3:
                    self.qubo[(i, j)] = self.intra_clause_weight
                # Punish contradictions
                elif abs(self.literals[i]) == abs(self.literals[j]) and self.literals[i] != self.literals[j]:
                    self.qubo[(i, j)] = self.inter_clause_weight

    def is_solution(self, solution_candidate: dict):
        """
        In this QUBO transformation, a "1" at position "i" means, that the literal at position "i" is part of the
        Maximum Independet Set. Therefore, if the i-th bit is set to 1 and the i-th bit represents literal -x it
        follows, that x:= 0.

        IMPORTANT: In CHOI's transformation it is possible that a solver finds contradictory solutions. There are
        multiple literals that represent the same variable, and their negation. Therefore, it is possible that
        a variable is set to 0 and 1 at the same time. This is a contradiction that must be resolved. As there is no
        method of knowing whether a contradictory variable is 0 or 1 - it is guessed.

        In this method guessing is done by overwriting. Each variable is set to 0 at first. Then we calculate all
        literals that are set to 1 by a solver - as the same variable (and its negation) is represented by multiple
        literals it is possible that the value of a variable is changed multiple times. Thus the value of a variable is
        either 0, if no literal representing this variable are set to 1 by a solver - or the value is 1 if the last
        literal representing this variable is set to 1 by a solver - or the value is 0 if the last literal representing
        this variable is set to 0 by a solver.

        Example: Suppose we are given the formula represented by [[1,2,3], [-1,2,3]] and a solver returns the solution
        1 = 1
        -1 = 1

        then x1 is assigned the 0, as the literal -1 is assigned the value "1" last.
        """
        true_indices = [literal_index for (literal_index, literal_value) in solution_candidate.items() if literal_value == 1]
        true_variables = list(set([self.literals[i] for i in true_indices]))

        assignment_dict = {abs(literal): 0 for literal in self.literals}
        assignment_dict.update({abs(literal): 1 for literal in true_variables})
        sat_clauses = self._get_satisfied_clauses(assignment_dict)


        # Check contradiction
        for variable in true_variables:
            if -variable in true_variables:
                return False, sat_clauses

        if sat_clauses == len(self.cnf.clauses):
            return True, sat_clauses
        else:
            return False, sat_clauses

    def print_qubo(self):
        var_names = {idx: "x"+str(self.literals[idx]) if self.literals[idx] > 0 else "-x"+str(abs(self.literals[idx]))
                     for idx in range(len(self.literals))}
        super()._print_qubo(var_names)


class ChancellorSAT(_QUBOTransformationBase):
    """
    This class creates a QUBO model for 3SAT problems, according to the special case (page 5)
    in Nick Chancellors's paper: https://www.nature.com/articles/srep37107.pdf.

    This implementation follows the Pattern QUBO principle. Each claus is sorted such that the negated variables
    are at the end of each clause. So Chancellors method only needs to be applied for type 0-3 clauses. (see
    https://www.mdpi.com/2079-9292/12/16/3492 for more information).

    This implementation uses the following values for the parameters of Chancellor's method:
    h=g=1 -> J_a = 2J > |h| --> choose J=h=1 -> J_a = 2, h_a = 2
    """

    def __init__(self, cnf: CNF):
        super().__init__(cnf)
        self.max_var = max(abs(min(self.cnf.var_list)), abs(max(self.cnf.var_list)))
        self.ancilla_list = []


    def create_qubo(self):
        for clause_index, clause in enumerate(self.cnf.clauses):
            if list(np.sign(clause)) == [1, 1, 1]:
                self.add(clause[0], clause[0], -2)
                self.add(clause[1], clause[1], -2)
                self.add(clause[2], clause[2], -2)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, -2)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], self.max_var + clause_index + 1, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, 1)

                self.add(clause[2], self.max_var + clause_index + 1, 1)

                self.ancilla_list.append(self.max_var + clause_index + 1)

            elif list(np.sign(clause)) == [1, 1, -1]:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], 0)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, -1)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], self.max_var + clause_index + 1, 1)

                self.add(clause[1], clause[2], 0)
                self.add(clause[1], self.max_var + clause_index + 1, 1)

                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.ancilla_list.append(self.max_var + clause_index + 1)

            elif list(np.sign(clause)) == [1, -1, -1]:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], -1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, -2)

                self.add(clause[0], clause[1], 0)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], self.max_var + clause_index + 1, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, 1)

                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.ancilla_list.append(self.max_var + clause_index + 1)

            else:
                self.add(clause[0], clause[0], -1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[2], clause[2], -1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, -1)

                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], self.max_var + clause_index + 1, 1)

                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, 1)

                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.ancilla_list.append(self.max_var + clause_index + 1)

    def is_solution(self, solution_candidate: dict):
        return self.cnf.is_satisfied(solution_candidate), self.cnf.count_satisfied_clauses(solution_candidate)

    def print_qubo(self):
        variable_labels = {abs(var): f"x{abs(var)}" for var in self.cnf.var_list}
        variable_labels.update({anc: f"{anc}(A)" for anc in self.ancilla_list})
        super()._print_qubo(variable_labels)


class Nuesslein2NMSAT(_QUBOTransformationBase):
    """
      This class provides an implementation of the 2nm sized QUBO transformation for 3SAT problems as described in the paper:
      https://dl.acm.org/doi/abs/10.1007/978-3-031-36030-5_3
      """

    def __init__(self, cnf: CNF, contradiction_penalty=None):
        super().__init__(cnf)
        self.literals = []

        if contradiction_penalty is None:
            self.contradiction_penalty = len(self.cnf.clauses) + 1
        else:
            self.contradiction_penalty = contradiction_penalty

        for var in cnf.abs_var_list:
            self.literals.append(var)
            self.literals.append(-var)

    def _r1(self, literal):
        n = 0
        for clause in self.cnf.clauses:
            if literal in clause:
                n += 1
        return n

    def _r2(self, literal1, literal2):
        n = 0
        for clause in self.cnf.clauses:
            if literal1 in clause and literal2 in clause:
                n += 1
        return n

    def create_qubo(self):
        for i in range(2*self.cnf.num_variables + len(self.cnf.clauses)):
            for j in range(2*self.cnf.num_variables + len(self.cnf.clauses)):
                if i > j:
                    continue
                if i == j and j < 2 * self.cnf.num_variables:
                    self.add(i, j, -self._r1(self.literals[i]))
                elif i == j and j >= 2 * self.cnf.num_variables:
                    self.add(i, j, 2)
                elif j < 2 * self.cnf.num_variables and j-i == 1 and i % 2 == 0:
                    self.add(i, j, self.contradiction_penalty)
                elif i < 2 * self.cnf.num_variables and j < 2*self.cnf.num_variables:
                    self.add(i, j, self._r2(self.literals[i], self.literals[j]))
                elif j >= 2*self.cnf.num_variables and i < 2 * self.cnf.num_variables and \
                        self.literals[i] in self.cnf.clauses[j - 2 * self.cnf.num_variables]:
                    self.add(i, j, -1)

    def is_solution(self, solution_candidate: dict):
        assignment = {var: 0 for var in self.cnf.abs_var_list}
        contradictions = 0
        for i in range(self.cnf.num_variables):
            k = 2 * i
            if solution_candidate[k + 1] == 1: # negative literal
                assignment.update({self.literals[k]: 0})
            elif solution_candidate[k + 1] == 0:
                assignment.update({self.literals[k]: 1})

            if solution_candidate[k] == solution_candidate[k+1]:
                contradictions += 1

        return self.cnf.is_satisfied(assignment), self.cnf.count_satisfied_clauses(assignment)



class NuessleinNMSAT(_QUBOTransformationBase):
    """
    This class provides an implementation of the nm-sized QUBO transformation for 3SAT problems as described in the paper:
    https://dl.acm.org/doi/abs/10.1007/978-3-031-36030-5_3
    """

    def __init__(self, cnf: CNF):
        super().__init__(cnf)
        self.max_var = max(self.cnf.abs_var_list)
        self.ancilla_list = []

    def create_qubo(self):
        for clause_index, clause in enumerate(self.cnf.clauses):
            if list(np.sign(clause)) == [1, 1, 1]:
                self.add(clause[0], clause[1], 2)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], -1)
                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 1)
                
                self.ancilla_list.append(self.max_var + clause_index + 1)
            elif list(np.sign(clause)) == [1, 1, -1]:
                self.add(clause[0], clause[1], 2)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], 1)
                self.add(clause[2], self.max_var + clause_index + 1, -1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 2)
                
                self.ancilla_list.append(self.max_var + clause_index + 1)
            elif list(np.sign(clause)) == [1, -1, -1]:
                self.add(clause[0], clause[0], 2)
                self.add(clause[0], clause[1], -2)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], self.max_var + clause_index + 1, 2)
                self.add(clause[2], clause[2], 1)
                self.add(clause[2], self.max_var + clause_index + 1, -1)

                self.ancilla_list.append(self.max_var + clause_index + 1)
            else:
                self.add(clause[0], clause[0], -1)
                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], self.max_var + clause_index + 1, 1)
                self.add(clause[1], clause[1], -1)
                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, 1)
                self.add(clause[2], clause[2], -1)
                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, -1)

                self.ancilla_list.append(self.max_var + clause_index + 1)

    def is_solution(self, solution_candidate: dict):
        return self.cnf.is_satisfied(solution_candidate), self.cnf.count_satisfied_clauses(solution_candidate)

    def print_qubo(self):
        variable_labels = {abs(var): f"x{abs(var)}" for var in self.cnf.var_list}
        variable_labels.update({anc: f"{anc}(A)" for anc in self.ancilla_list})
        super()._print_qubo(variable_labels)


class RosenbergSAT(_QUBOTransformationBase):
    """
    This class provides an implementation of the nm-sized QUBO transformation for 3SAT problems as discovered by
    Rosenberg. This QUBO transformation is often referenced in literature see for example:

    https://arxiv.org/pdf/2107.11695
    https://eprints.lse.ac.uk/66580/1/__lse.ac.uk_storage_LIBRARY_Secondary_libfile_shared_repository_Content_Anthony,%20M_Quadratic%20reformulations%20of%20nonlinear%20binary_Anthony_Quadratic_reformulations_of_nonlinear_binary.pdf

    """

    def __init__(self, cnf: CNF):
        super().__init__(cnf)
        self.max_var = max(self.cnf.abs_var_list)
        self.ancilla_list = []

    def create_qubo(self):
        for clause_index, clause in enumerate(self.cnf.clauses):
            if list(np.sign(clause)) == [1, 1, 1]:
                self.add(clause[0], clause[0], -1)
                self.add(clause[0], clause[1], 2)
                self.add(clause[0], clause[2], 1)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], clause[1], -1)
                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], -1)
                self.add(clause[2], self.max_var + clause_index + 1, -1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 3)

                self.ancilla_list.append(self.max_var + clause_index + 1)
            elif list(np.sign(clause)) == [1, 1, -1]:
                self.add(clause[0], clause[0], 0)
                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], -1)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], clause[1], 0)
                self.add(clause[1], clause[2], -1)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], 1)
                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 3)

                self.ancilla_list.append(self.max_var + clause_index + 1)
            elif list(np.sign(clause)) == [1, -1, -1]:
                self.add(clause[0], clause[0], 0)
                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], clause[1], 0)
                self.add(clause[1], clause[2], 1)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], 0)
                self.add(clause[2], self.max_var + clause_index + 1, -1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 3)

                self.ancilla_list.append(self.max_var + clause_index + 1)
            else:
                self.add(clause[0], clause[0], 0)
                self.add(clause[0], clause[1], 1)
                self.add(clause[0], clause[2], 0)
                self.add(clause[0], self.max_var + clause_index + 1, -2)
                self.add(clause[1], clause[1], 0)
                self.add(clause[1], clause[2], 0)
                self.add(clause[1], self.max_var + clause_index + 1, -2)
                self.add(clause[2], clause[2], 0)
                self.add(clause[2], self.max_var + clause_index + 1, 1)
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, 3)

                self.ancilla_list.append(self.max_var + clause_index + 1)

    def is_solution(self, solution_candidate: dict):
        return self.cnf.is_satisfied(solution_candidate), self.cnf.count_satisfied_clauses(solution_candidate)

    def print_qubo(self):
        variable_labels = {abs(var): f"x{abs(var)}" for var in self.cnf.var_list}
        variable_labels.update({anc: f"{anc}(A)" for anc in self.ancilla_list})
        super()._print_qubo(variable_labels)


# ------------------------------------------------------------


class _QUBOIterator:
    """
    Iterator, that generates all possible QUBO matrices for a given QUBO size and admissible value set.
    The iterator is used in the PatternQUBOFinder class, where it will be used in a parallel, multiprocess environment.
    An iterator is needed, to avoid memory issues, when generating all possible QUBO matrices.

    A QUBO matrix is an upper triangular matrix -> here we represent a uppter triangular matrix as a list. The first row
    of the upper triangular matrix correspond to the first entries of the list, and so on). A 4x4 upper triangular
    QUBO matrix thus corresponds to a list of length 10 (first row of the QUBO matrix has 4 elemtens, second row has 3
    elements, thrid row has 2 elemtents and the last row has 1 element -> 4+3+2+1 = 10).
    """
    def __init__(self, qubo_size, admissible_values):
        self.overflow = max(admissible_values) + 1
        self.qubo_size = qubo_size
        self.admissible_values = sorted(admissible_values) + [self.overflow]
        self.current_qubo = [min(self.admissible_values)] * self.qubo_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_qubo == [self.admissible_values[-2]] * len(self.current_qubo):
            raise StopIteration

        current_qubo = deepcopy(self.current_qubo)
        self._update_qubo()
        return current_qubo

    def _update_qubo(self):
        self.current_qubo[-1] = self.admissible_values[self.admissible_values.index(self.current_qubo[-1]) + 1]

        for i in range(len(self.current_qubo)):
            if self.current_qubo[-1 - i] == self.overflow:
                self.current_qubo[-1 - i] = self.admissible_values[0]
                self.current_qubo[-1 - (i + 1)] = self.admissible_values[self.admissible_values.index(self.current_qubo[-1 - (i + 1)]) + 1]

        return self.current_qubo


class PatternQUBOFinder:
    """
    This class provides a multiprocess implementation of the Pattern QUBO Method,as described in the paper:
    https://www.mdpi.com/2079-9292/12/16/3492
    """

    def __init__(self, num_parallel_processes):
        self.num_parallel_processes = num_parallel_processes

    def find(self, admissible_values_set):

        pattern_qubo_list = self._find_pattern_qubos(admissible_values_set)

        pattern_qubo_dict = {}
        for i in range(4):
            pattern_qubo_dict[i] = []

        for clause_type, pattern_qubo in pattern_qubo_list:
            pattern_qubo_dict[clause_type].append(self._transform_qubo_list_to_dict(pattern_qubo, 4))

        return pattern_qubo_dict

    def _find_pattern_qubos(self, admissible_values_set):
        qubo_iterator = _QUBOIterator(qubo_size=10, admissible_values=admissible_values_set)
        results = mp.Manager().list()
        worker_partial = functools.partial(self._filter_qubos)

        with Pool(self.num_parallel_processes) as p:
            for result in p.imap(worker_partial, qubo_iterator):
                if result[0]:
                    results.append((result[1], result[2]))
        return results

    def _filter_qubos(self, qubo):

        solutions = {}
        num_solutions = 16  # 4x4 QUBO, 2^4 possible solutions

        for solution in range(0, num_solutions):
            sample = [int(j) for j in f'{solution:0{4}b}']

            sample_energy = self._qubo_energy(qubo, sample)

            # 4x4 QUBO, 3 variables -> 2^3 = 8 possible solutions, 1 ancilla: each solution corresponds to 2 samples:
            # i.e. solution 1 0 0 corresponds to samples 1 0 0 0 and 1 0 0 1, where the last bit is the value of the
            # ancilla bit

            solution_key = solution >> 1
            current_energy = solutions.setdefault(solution_key, sample_energy)
            if current_energy > sample_energy:
                solutions[solution_key] = sample_energy

        values = list(solutions.values())

        # each clause has 7 correct assignments
        if values.count(min(values)) != 7:
            return False,
        is_correct, clause_type = self._correct_mn_qubo_filter(solutions)
        return is_correct, clause_type, qubo

    def _correct_mn_qubo_filter(self, solutions):

        always_correct = solutions[5]

        if solutions[0] > always_correct:
            return True, 0
        if solutions[1] > always_correct:
            return True, 1
        if solutions[3] > always_correct:
            return True, 2
        if solutions[7] > always_correct:
            return True, 3

        return False, None

    def _qubo_energy(self, qubo, sample):
        energy = 0
        idx = 0
        sample_length = len(sample)
        for i in range(sample_length):
            for j in range(i, sample_length):
                energy += qubo[idx] * sample[i] * sample[j]
                idx += 1
        return energy

    def _transform_qubo_list_to_dict(self, qubo, size):
        qubo_dict = {}
        idx = 0
        for i in range(size):
            for j in range(i, size):
                qubo_dict[(i, j)] = qubo[idx]
                idx += 1
        return qubo_dict

    @staticmethod
    def save(pattern_qubos: dict, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(pattern_qubos, file)

    @staticmethod
    def load(abs_file_path):
        with open(abs_file_path, "rb") as file:
            return pickle.load(file)


class PatternQUBONMSAT(_QUBOTransformationBase):
    """
    This class is used to create a concrete QUBO transformation, based on the Pattern QUBO Method.
    By providing Pattern QUBOs for each type of the four types of  clause (type 0 - 3), the QUBO matrix can be
    obtained by following the Pattern QUBO constrution as presented in:  https://www.mdpi.com/2079-9292/12/16/3492

    Pattern QUBOs can be obtained by using the PatternQUBOFinder class.
    """
    def __init__(self, cnf=CNF([])):
        super().__init__(cnf)
        self.clause_qubos = []
        if len(self.cnf.clauses) != 0:
            self.max_var = max(self.cnf.abs_var_list)
        self.ancilla_list = []

    def create_qubo(self):
        for clause_index, clause in enumerate(self.cnf.clauses):
            if list(np.sign(clause)) == [1, 1, 1]:
                self.add(clause[0], clause[0], self.clause_qubos[0][(0, 0)])
                self.add(clause[0], clause[1], self.clause_qubos[0][(0, 1)])
                self.add(clause[0], clause[2], self.clause_qubos[0][(0, 2)])
                self.add(clause[0], self.max_var + clause_index + 1, self.clause_qubos[0][(0, 3)])
                self.add(clause[1], clause[1], self.clause_qubos[0][(1, 1)])
                self.add(clause[1], clause[2], self.clause_qubos[0][(1, 2)])
                self.add(clause[1], self.max_var + clause_index + 1, self.clause_qubos[0][(1, 3)])
                self.add(clause[2], clause[2], self.clause_qubos[0][(2, 2)])
                self.add(clause[2], self.max_var + clause_index + 1, self.clause_qubos[0][(2, 3)])
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, self.clause_qubos[0][(3, 3)])

                self.ancilla_list.append(self.max_var + clause_index + 1)

            elif list(np.sign(clause)) == [1, 1, -1]:
                self.add(clause[0], clause[0], self.clause_qubos[1][(0, 0)])
                self.add(clause[0], clause[1], self.clause_qubos[1][(0, 1)])
                self.add(clause[0], clause[2], self.clause_qubos[1][(0, 2)])
                self.add(clause[0], self.max_var + clause_index + 1, self.clause_qubos[1][(0, 3)])
                self.add(clause[1], clause[1], self.clause_qubos[1][(1, 1)])
                self.add(clause[1], clause[2], self.clause_qubos[1][(1, 2)])
                self.add(clause[1], self.max_var + clause_index + 1, self.clause_qubos[1][(1, 3)])
                self.add(clause[2], clause[2], self.clause_qubos[1][(2, 2)])
                self.add(clause[2], self.max_var + clause_index + 1, self.clause_qubos[1][(2, 3)])
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, self.clause_qubos[1][(3, 3)])

                self.ancilla_list.append(self.max_var + clause_index + 1)

            elif list(np.sign(clause)) == [1, -1, -1]:
                self.add(clause[0], clause[0], self.clause_qubos[2][(0, 0)])
                self.add(clause[0], clause[1], self.clause_qubos[2][(0, 1)])
                self.add(clause[0], clause[2], self.clause_qubos[2][(0, 2)])
                self.add(clause[0], self.max_var + clause_index + 1, self.clause_qubos[2][(0, 3)])
                self.add(clause[1], clause[1], self.clause_qubos[2][(1, 1)])
                self.add(clause[1], clause[2], self.clause_qubos[2][(1, 2)])
                self.add(clause[1], self.max_var + clause_index + 1, self.clause_qubos[2][(1, 3)])
                self.add(clause[2], clause[2], self.clause_qubos[2][(2, 2)])
                self.add(clause[2], self.max_var + clause_index + 1, self.clause_qubos[2][(2, 3)])
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, self.clause_qubos[2][(3, 3)])

                self.ancilla_list.append(self.max_var + clause_index + 1)

            else:
                self.add(clause[0], clause[0], self.clause_qubos[3][(0, 0)])
                self.add(clause[0], clause[1], self.clause_qubos[3][(0, 1)])
                self.add(clause[0], clause[2], self.clause_qubos[3][(0, 2)])
                self.add(clause[0], self.max_var + clause_index + 1, self.clause_qubos[3][(0, 3)])
                self.add(clause[1], clause[1], self.clause_qubos[3][(1, 1)])
                self.add(clause[1], clause[2], self.clause_qubos[3][(1, 2)])
                self.add(clause[1], self.max_var + clause_index + 1, self.clause_qubos[3][(1, 3)])
                self.add(clause[2], clause[2], self.clause_qubos[3][(2, 2)])
                self.add(clause[2], self.max_var + clause_index + 1, self.clause_qubos[3][(2, 3)])
                self.add(self.max_var + clause_index + 1, self.max_var + clause_index + 1, self.clause_qubos[3][(3, 3)])

                self.ancilla_list.append(self.max_var + clause_index + 1)

    def is_solution(self, solution_candidate: dict):
        return self.cnf.is_satisfied(solution_candidate), self.cnf.count_satisfied_clauses(solution_candidate)

    def print_qubo(self):
        variable_labels = {abs(var): f"x{abs(var)}" for var in self.cnf.var_list}
        variable_labels.update({anc: f"{anc}(A)" for anc in self.ancilla_list})
        super()._print_qubo(variable_labels)

    def add_clause_qubos(self, c0, c1, c2, c3):
        self.clause_qubos = [c0, c1, c2, c3]

    def export(self, filename, transformation_name="PatternQUBONM"):
        """
        This function creates a standalone QUBO-Transformation class, that does not depend on SATQUBOLIB anymore.
        """
        import inspect

        transformation_string = inspect.getsource(PatternQUBONMSAT)
        transformation_string = transformation_string.replace("PatternQUBONMSAT", transformation_name)

        for clause_type in range(4):
            for i in range(4):
                for j in range(i, 4):
                    transformation_string = transformation_string.replace(
                        "self.clause_qubos[" + str(clause_type) + "][(" + str(i) + ", " + str(j) + ")]",
                        str(self.clause_qubos[clause_type].get((i,j), 0))
                    )

        with open(filename, "w") as file:
            file.write("import numpy as np\n")
            file.write("import warnings\n")
            file.write(inspect.getsource(CNF))
            file.write("\n\n")
            file.write(inspect.getsource(_QUBOTransformationBase))
            file.write("\n\n")
            file.write("\n".join(transformation_string.split("\n")[:-31]))

        print(f"Transformation successfully exported to {filename}")
