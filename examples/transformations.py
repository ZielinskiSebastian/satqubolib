"""
This module documents and explains the satqubolib.transformations module.

The goal of a transformation is to create QUBO representations of CNF formulas. Different transformations result in
different QUBOs that may produce significantly different outcomes when solved by QUBO solvers. In this module, we will
demonstrate how to use the different transformations provided by the satqubolib library.
"""
from dwave.system import DWaveSampler, EmbeddingComposite

from satqubolib.formula import CNF
from satqubolib.transformations import ChoiSAT, ChancellorSAT, NuessleinNMSAT, Nuesslein2NMSAT, PatternQUBOFinder, PatternQUBONMSAT, RosenbergSAT
from tabu import TabuSampler
from dimod.binary_quadratic_model import BinaryQuadraticModel

"""
Example 1 - Using a predefined transformation
"""


def example_1():
    # satqubolib provides explicit implementations of some well known QUBO transformations, as well as implicit
    # implementations of further thousands of transformations. In this example we will introduce the usage of the explicit
    # transformations: ChoiSAT, ChancellorSAT, and NuessleinNMSAT, Nuesslein2NMSAT, RosenbergSAT

    # Let's start by creating a CNF formula - for further information on CNF formulas, please refer to the
    # examples.formula module
    cnf = CNF([[1, 2, 3], [1, 2, -3]])  # The CNF formula is (x1 v x2 v x3) and (x1 v x2 v -x3)

    sat_qubo = ChoiSAT(cnf)  # Hand over a CNF formula to the transformation object
    # sat_qubo = ChancellorSAT(cnf)
    # sat_qubo = NuessleinNMSAT(cnf)
    # sat_qubo = Nuesslein2NMSAT(cnf)
    # sat_qubo = RosenbergSAT(cnf)
    sat_qubo.create_qubo()  # Execute the transformation algorithm to create the QUBO matrix

    # For smaller formulas, the print_qubo() function is useful to display the QUBO matrix
    sat_qubo.print_qubo()


"""
Example 2 - Verifying a solution to a QUBO
"""


def example_2():
    # Let's start by creating a CNF formula - for further information on CNF formulas, please refer to the
    # examples.formula module
    cnf = CNF([[1, 2, 3], [1, 2, -3]])  # The CNF formula is (x1 v x2 v x3) and (x1 v x2 v -x3)

    sat_qubo = ChoiSAT(cnf)  # Sample Solution: {0: 0, 1:0, 2:1, 3:0, 4:0, 5:1} - Will return (False, 1), only second clause is satisfied
    # sat_qubo = ChancellorSAT(cnf) # Sample Solution: {1:0, 2:1, 3:0} - Will return (True, 2), both clauses are satisfied
    # sat_qubo = NuessleinNMSAT(cnf) # Sample Solution: {1:0, 2:1, 3:0} - Will return (True, 2), both clauses are satisfied
    # sat_qubo = Nuesslein2NMSAT(cnf) # Sample Solution: {1:0, 2:1, 3:0} - Will return (True, 2), both clauses are satisfied
    # sat_qubo = RosenbergSAT(cnf) # Sample Solution: {1:0, 2:1, 3:0} - Will return (True, 2), both clauses are satisfied

    sat_qubo.create_qubo()  # Execute the transformation algorithm to create the QUBO matrix

    # Given one of the sample solution candidates above, we can check if the solution candidate is valid
    # solution by using the is_solution() function:

    is_solution = sat_qubo.is_solution({1: 0, 2: 1, 3: 0})
    print(is_solution)  # Returns a tuple (Boolean, Int). The Boolean states whether the solution is valid, the Int states the number of satisfied clauses

###################################
# IMPORTANT NOTE: The satqubolib.formula.CNF class also provides a is_solution() funtion.
# DO NOT use the cnf.is_solution() function, to check whether a solution candidate returned by a QUBO solver
# like D-Wave's Quantum Annealer, D-Wave's Tabu Search, Simulated Annealing, ... is a valid solution to the CNF.
# Each QUBO transformations performs a custom mapping of CNF variables to QUBO variables, hence the
# is_solution() method of the CNF object will not work correctly in this context. In this case the
# is_solution() method of the transformation object must be used!
###################################

"""
Example 3 - Using the PatternQUBOFinder to search Pattern QUBOs
"""

# In this example we will demonstrate how to use the PatternQUBOFinder to search for valid pattern qubos for all
# types of clauses. In the next example we will use these pattern qubos, to create thousands of new QUBO transformations.

def example_3():
    # First we create a PatternQUBOFinder object that will perform the pattern qubo search using 20 parallel threads.
    # Increase / Decrease this value as you please / according to your system specs.
    pqf = PatternQUBOFinder(20)
    # Specify a set of values the pattern qubo method is allowed to fill into the QUBO matrix.
    # The set does not have to be symmetric - any set of values can be used. However, depending on the set of numbers
    # sepcified, there may not exist any valid pattern qubos.
    # NOTE: as the pattern qubo method is exhaustive search, the search duration increases exponentially as the set of
    # admissible values grows.

    admissible_values = {-1, 0, 1}

    # Now we perform the search for pattern qubos. The search will return a dictionary containing the found pattern qubos
    # IMPORTANT: As the search is using multiprocessing to speed up the search, this function call must by from within an
    # if __name__ == "__main__": block. Otherwise, the multiprocessing module will not work correctly.
    pattern_qubos = pqf.find(admissible_values)

    # After performing the  search, we save the found pattern qubos to a file for later use. satqubolib uses pickle to
    # serialize the data
    pqf.save(pattern_qubos, "pattern_qubos.pkl")

    # Later, we can load the pattern qubos from the file:
    pattern_qubos = pqf.load("pattern_qubos.pkl")

    # Now let us understand the pattern_qubos object we created. The pattern_qubos object is a dictionary containing four
    # keys: 0, 1, 2, 3. Each key corresponds to a SAT clause type. Type 0 means the clause has 0 negations. Type 1 means,
    # the clause has 1 negation, and so on. Each key contains a list of valid pattern qubos for the respective clause type.

    for key in pattern_qubos:
        print(f"Clause type: {key}\nNumber of Pattern QUBOs: {len(pattern_qubos[key])}\nPattern QUBOs: {pattern_qubos[key]}")
        print("----------")

    # By executing this example you can see, that the pattern qubo method finds a different number of pattern qubos
    # for each clause type. The number of pattern qubos found depends on the set of admissible values specified in the
    # find() method.

"""
Example 4 - Creating new QUBO transformations using the found Pattern QUBOs
"""

def example_4():
    # In this example we will demonstrate how to use the found pattern qubos to create new QUBO transformations.
    # The new QUBO transformations will be created for a given CNF formula, using the found pattern qubos.

    # First we create a CNF formula - for further information on CNF formulas, please refer to the examples.formula module
    cnf = CNF([[1, 2, 3], [1, 2, -3]])  # The CNF formula is (x1 v x2 v x3) and (x1 v x2 v -x3)

    # Next we create a PatternQUBOFinder object that will perform the pattern qubo search using 20 parallel threads.
    # Increase / Decrease this value as you please / according to your system specs.
    pqf = PatternQUBOFinder(20)

    # Now we load the pattern qubos from a file, that we have previously saved.
    pattern_qubos = pqf.load("pattern_qubos.pkl")

    # Now we create a new QUBO transformation object, and add the found pattern qubos to the transformation
    sat_qubo = PatternQUBONMSAT(cnf)
    # To create a new QUBO transformation, we need to add one  pattern qubo for each clause type to the transformation
    # NOTE: Using the admissible value set {-1, 0, 1} we found 6 pattern qubos for clauses of type 0, 7 pattern qubos
    # for clauses of type 1, 6 pattern qubos for clauses of type 2, and 8 pattern qubos for clauses of type 3.
    # Thus, there are 6 * 7 * 6 * 8 = 2016 possible QUBO transformations that can be created using the found pattern qubos.
    sat_qubo.add_clause_qubos(pattern_qubos[0][0], pattern_qubos[1][0], pattern_qubos[2][0], pattern_qubos[3][0])
    sat_qubo.create_qubo()  # Execute the transformation algorithm to create the QUBO matrix
    sat_qubo.print_qubo()


"""
Example 5: Exporting a QUBO transformation to a standalone python file
"""


def example_5():
    # In this example we demonstrate how to export a QUBO transformation to a standalone python file.

    # We create a PatternQUBOFinder object that will perform the pattern qubo search using 20 parallel threads.
    # Increase / Decrease this value as you please / according to your system specs.
    pqf = PatternQUBOFinder(20)


    # Now we load the pattern qubos from a file, that we have previously saved.
    pattern_qubos = pqf.load("pattern_qubos.pkl")

    # Now we create a new QUBO transformation object, and add the found pattern qubos to the transformation
    sat_qubo = PatternQUBONMSAT()
    # To create a new QUBO transformation, we need to add one  pattern qubo for each clause type to the transformation
    sat_qubo.add_clause_qubos(pattern_qubos[0][0], pattern_qubos[1][0], pattern_qubos[2][0], pattern_qubos[3][0])

    # Finally we export the transformation to a standalone python file
    sat_qubo.export(filename="myqubo.py", transformation_name="MyQUBO")

    # This will create a qubo transformation called "MyQUBO" in a file called "myqubo.py". It can be used as any
    # other qubo transformation in the satqubolib library (i.e. as demonstrated in the examples 1 and 2 above).
    # from myqubo import MyQUBO
    # sat_qubo = MyQUBO(CNF([[1, 2, 3], [1, 2, -3]]))
    # sat_qubo.create_qubo()
    # sat_qubo.print_qubo()


"""
Example 6: Solving a QUBO using TABU-Search
"""


def example_6():
    cnf = CNF([[1, 4, 7], [2,5,87]])  # The CNF formula is (x1 v x2 v x3) and (x1 v x2 v -x3)
    # sat_qubo = ChoiSAT(cnf)
    sat_qubo = ChancellorSAT(cnf)
    # sat_qubo = NuessleinNMSAT(cnf)
    # sat_qubo = Nuesslein2NMSAT(cnf)

    sat_qubo.create_qubo()
    # sat_qubo.print_qubo()
    # Now we create a BinaryQuadraticModel from the QUBO matrix
    # Documentation: https://docs.ocean.dwavesys.com/en/latest/docs_dimod/reference/models.html#module-dimod.binary.binary_quadratic_model
    bqm = BinaryQuadraticModel.from_qubo(sat_qubo.qubo)

    # Create a TabuSampler object that is used to perform the tabu search
    sampler = TabuSampler()
    # Perform the search using the tabu search algorithm, and return the solutions in a sampleset.
    # num_reads specifies the number of samples to be generated by the sampler. For further information
    # refer to the documentation of the TabuSampler class: https://docs.ocean.dwavesys.com/projects/tabu/en/latest/reference/sampler.html
    sampleset = sampler.sample(bqm, num_reads=10, answer_mode="histogram")

    # The sampleset object contains the results of the sampling. It is a dictionary-like object that contains the
    # samples, energies, and other information about the sampling process.
    # Documentation: https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html

    print(sampleset, "\n")

    # To access individual samples from the sampleset, we use the samples() method:

    for index, sample in enumerate(sampleset.samples()):
        sample = dict(sample) #  First transform the sample from a dimod.SampleView object to a dictionary
        print("Solution #"+str(index)+": ", sat_qubo.is_solution(sample))


"""
Example 7: Solving a QUBO using DWave's Quantum Annealer
"""

# To execute this example, you need your own D-Wave API token.

def example_7():
    cnf = CNF([[1, 2, 3], [1, 2, -3]])  # The CNF formula is (x1 v x2 v x3) and (x1 v x2 v -x3)

    sat_qubo = ChoiSAT(cnf)
    # sat_qubo = ChancellorSAT(cnf)
    # sat_qubo = NuessleinNMSAT(cnf)
    # sat_qubo = Nuesslein2NMSAT(cnf)
    # sat_qubo = RosenbergSAT(cnf)

    sat_qubo.create_qubo()
    # Now we create a BinaryQuadraticModel from the QUBO matrix
    # Documentation: https://docs.ocean.dwavesys.com/en/latest/docs_dimod/reference/models.html#module-dimod.binary.binary_quadratic_model
    bqm = BinaryQuadraticModel.from_qubo(sat_qubo.qubo)

    # Create a DWaveSampler object that is used to perform the quantum annealing
    # Documentation: https://docs.ocean.dwavesys.com/projects/system/en/latest/reference/samplers.html
    sampler = DWaveSampler(endpoint="https://cloud.dwavesys.com/sapi", solver="Advantage_system6.4", token="YOUR_TOKEN")
    # Qubits on current D-Wave quantum annealers are not all to all connected. Thus to solve a qubo,
    # a mapping from logical variables to the physical qubits need to be found - this process is calld (minor) embedding.
    # The EmbeddingComposite object is used to perform the embedding.
    # Documentation: https://docs.ocean.dwavesys.com/projects/system/en/latest/reference/composites.html#embeddingcomposite
    sampler = EmbeddingComposite(sampler)
    sampleset = sampler.sample(bqm, num_reads=10, answer_mode="histogram", return_embedding=True)

    # The sampleset object contains the results of the sampling. It is a dictionary-like object that contains the
    # samples, energies, and other information about the sampling process.
    # Documentation: https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html

    print(sampleset, "\n")

    # To access individual samples from the sampleset, we use the samples() method:

    for index, sample in enumerate(sampleset.samples()):
        sample = dict(sample)  # First transform the sample from a dimod.SampleView object to a dictionary
        print("Solution #" + str(index) + ": ", sat_qubo.is_solution(sample))

    # D-Wave provides additional information regarding embedding end timing. This information can be accessed using the
    # info attribute of the sampleset object.

    print("\n", sampleset.info)


if __name__ == "__main__":
    example_1()
    # example_2()
    # example_3()
    # example_4()
    # example_5()
    # example_6()
    # example_7()
