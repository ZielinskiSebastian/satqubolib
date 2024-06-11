import warnings


class CNF:

    def __init__(self, clauses: list):
        """
                Any clause within a formula in conjunctive normal form (CNF) specified by its clauses :clauses:
                will be sorted according to the following rules:
                - positive literals are at the beginning of the clause, negative literals at the end
                - positive literals will be sorted in ascending order (i.e. smallest to largest)
                - negative literals will be sorted in descending order (i.e. largest to smallest)

                i.e.
                    clause [4, 1, -3] will be sorted to [1, 4, -3]
                    clause [1, -4, -3] will be sorted to [1, -3, -4]
        """
        self.clauses = [sorted(clause, key=lambda value: -1 / value if value > 0 else -value) for clause in clauses]
        self.var_list = None
        self.abs_var_list = None
        self.num_variables = None
        self._update_variables()

        self.solution = None
        self.solve_time = None
        self.solver = None
        self.cpu = None

    def __str__(self):
        return str(self.clauses)

    def is_satisfied(self, assignment):
        return self.count_satisfied_clauses(assignment) == len(self.clauses)
    
    def count_satisfied_clauses(self, assignment):
        satisfied_clauses = 0
        for clause in self.clauses:
            for literal in clause:
                if literal < 0 and assignment[abs(literal)] == 0:
                    satisfied_clauses += 1
                    break
                elif literal > 0 and assignment[abs(literal)] == 1:
                    satisfied_clauses += 1
                    break
        return satisfied_clauses

    def add_clause(self, clause):
        self.clauses.append(clause)
        self._update_variables()

    def remove_clause(self, clause):
        try:
            self.clauses.remove(clause)
            self._update_variables()
        except ValueError:
            warnings.warn("Warning: Clause not found in formula. Nothing was removed.")
            pass

    def _update_variables(self):
        self.var_list = sorted(list(set([literal for clause in self.clauses for literal in clause])))
        self.abs_var_list = sorted(list(set([abs(literal) for clause in self.clauses for literal in clause])))
        self.num_variables = len(self.abs_var_list)

    @staticmethod
    def from_file(abs_path):
        """
        Load a CNF formula from a file in DIMACS format :abs_path:
        """

        solution = None
        solve_time = None
        solver = None
        cpu = None
        clauses = []

        with open(abs_path, "r") as file:
            for line in file:
                if line.startswith("c"):
                    line = line.split(" ")
                    if line[1].startswith("Solution"):
                        try:
                            solution = list(map(int, line[2:]))
                        except ValueError as ve:
                            raise Exception("Warning: Malformed solution in input file. Lines should not contain double blank spaces or a blank space before '\\n'.") from ve

                    elif line[1].startswith("Time"):
                        solve_time = float(line[2].replace("\n", "").replace(",", "."))
                    elif line[1].startswith("Solver"):
                        solver =  " ".join(line[2:]).replace("\n", "")
                    elif line[1].startswith("CPU"):
                        cpu = " ".join(line[2:]).replace("\n", "")
                    else:
                        pass
                elif not line.startswith("p"):
                    clause = list(map(lambda x: int(x), line.split(" ")))
                    # Discard DIMACS line end delimiter "0"
                    clause.remove(0)
                    if len(clause) < 3:
                        raise Exception("Detected clause with less than 3 literals. SATQUBOLIB currently only supports problems consisting of exactly 3 literals per clause. Clause: " + str(clause))
                    if 0 in clause:
                        raise Exception("Detected \"0\" in clause. DIMACS format does not allow for 0 literals. Clause: " + str(clause))
                    clauses.append(list(map(lambda x: int(x), line.split(" ")))[:-1])

        cnf = CNF(clauses)
        cnf.solution = solution
        cnf.solve_time = solve_time
        cnf.solver = solver
        cnf.cpu = cpu
        return cnf

    def to_file(self, abs_path):
        """
        Save the CNF formula to the file specified by :abs_path: using DIMACS format
        """
        with open(abs_path, "w+") as file:
            if self.solution is not None:
                file.write("c Solution: " + " ".join(map(str, self.solution)) + "\n")
            if self.solve_time is not None:
                file.write("c Time: " + str(self.solve_time) + " seconds \n")
            if self.solver is not None:
                file.write("c Solver: " + self.solver + " \n")
            if self.cpu is not None:
                file.write("c CPU: " + str(self.cpu) + " \n")
            file.write(f"p cnf {self.num_variables} {len(self.clauses)}\n")
            for clause in self.clauses:
                file.write(" ".join(map(str, clause)) + " 0\n")
