# Essential imports for the calculations
from qiskit import QuantumCircuit, Aer, transpile, ClassicalRegister
from qiskit.visualization import plot_gate_map, plot_error_map, plot_histogram
from qiskit.providers.fake_provider import FakeQuitoV2
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit.providers.fake_provider import FakeCasablancaV2
from qiskit.quantum_info.operators.symplectic import Pauli
from matplotlib import pyplot as plt
import sys
import numpy as np
import graphviz
import time 
sys.path.append("/home/prachi/Documents/Code_PER/AutomatedPERTools-main/pauli_lindblad_per/")
from tomography.experiment import SparsePauliTomographyExperiment as tomography
from primitives.pauli import QiskitPauli

from circuit_builder import *
from circuit_builder import CircuitBuilder
from model_hamiltonian import circuit_optimized_parameters
plt.style.use("ggplot")

start_time = time.time()

## PER functions and steps 

#class Noise_tomography(Quantum_system):
#    def __init__(self, circuit, pair_samples, single_samples, depth):
#        self.circuit = circuit
#        self.pair_samples = pair_samples
#        self.single_samples = single_samples
#        self.depth = depth
#        super.__init__()
#
#    def tomography_step(self):#, pair_samples = 1,single_samples =1, depth = [2,4,8,16]):
#        experiment = tomography(circuits = self.circuits, inst_map = self.initial_layout, backend = self.backend) # add initial layout as a parameter here
#        # to generate the circuits of different depths
#        experiment.generate(samples = self.pair_samples, single_samples = self.single_samples, depths = self.depth)
#        #run the experiment
#        experiment.run(executor)
#        noisedataframe = experiment.analyze()
#        return experiment, noisedataframe
    
class error_mitigation():
    def __init__(self,experiment, expectations, samples, noise_strengths, circuit):
        self.circuit = circuit
        self.experiment = experiment
        self.expectations = expectations
        self.samples = samples
        self.noise_strengths = noise_strengths


    def PER_step(self):
        perexp = experiment.create_per_experiment(self.circuit)
        perexp.generate(expectations = self.expectations, samples = self.samples, noise_strengths = self.noise_strengths)
        print(perexp.meas_bases)
        perexp.run(executor)
        circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
        #and apply vZNE
        return circuit_results


def tomography_step( circuits = None, pair_samples = 1,single_samples =1, depths = [2,4,8,16]
                    , backend = FakeQuitoV2(), initial_layout = [],shots = 1000):
    experiment = tomography(circuits = circuits, inst_map = initial_layout, backend = backend) # add initial layout as a parameter here
    # to generate the circuits of different depths
    experiment.generate(samples = pair_samples, single_samples = single_samples, depths =depths)
    #run the experiment
    help = helper_class(backend=backend)
    experiment.run(help.executor)
    noisedataframe = experiment.analyze()
    return experiment, noisedataframe

def PER_step(experiment = None, expectations = [], samples = 10, noise_strengths = [0, 0.25, 0.5],
             circuits = None,backend = FakeQuitoV2(), initial_layout = [],shots = 1000):
    perexp = experiment.create_per_experiment(circuits)
    perexp.generate(expectations = expectations, samples = samples, noise_strengths = noise_strengths)
    print(perexp.meas_bases)
    help = helper_class(backend=backend)
    perexp.run(help.executor)#(circuits = circuits,backend = backend, shots = shots))
    circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
    #and apply vZNE
    return circuit_results

# Make a helper.py and put such functions there
class helper_class(Quantum_system):
    def __init__(self,backend):
        super().__init__(backend)
    def executor(self, circuits, shots = 1000):
        return self.backend.run(circuits, shots = shots).result().get_counts()

#shots = 1000

def executor(circuits, backend = FakeQuitoV2(),shots = 1000):
        return backend.run(circuits, shots = shots).result().get_counts()

# TODO SV calculations

class QSimulator():
    def __init__(self,backend):
        # self.simulator = simulator
        self.backend = backend
        #self.circuits = circuits
        #self.shots = shots
        
    def State_Vector_Simulator(self,circuits):

        # The circuits here are without measurements
        count = Aer.get_backend('statevector_simulator').run(circuits[0]).result().get_statevector()
        return count

    def QASM_Simulator(self,circuits,shots=1000):

        # For Z measurements
        count_QASM_Z =Aer.get_backend('qasm_simulator').run(circuits[0], shots=shots).result().get_counts()
        # For X measurements
        count_QASM_X =Aer.get_backend('qasm_simulator').run(circuits[1], shots=shots).result().get_counts()
        return count_QASM_Z,count_QASM_X

    def Noisy_backend_Simulato(self,circuits,shots=1000):
        count_Z = self.backend.run(circuits[0], shots=shots).result().get_counts()
        count_X = self.backend.run(circuits[1], shots=shots).result().get_counts()
        return count_Z, count_X



def main():
  
    backend_configuration = Quantum_system(backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")  #FakeGuadalupeV2())
    opt_params_Quito =  circuit_optimized_parameters("FakeQuitoV2")
    circuits = CircuitBuilder(params = opt_params_Quito, backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")
    #print("Backend =", backend)
    #Circuits sent to tomography has to be a list
    circuits_w_no_meas = [circuits.makevqeCircuit(measure = False)]
    print(circuits_w_no_meas[0].draw())
    #print(type(circuits_w_no_meas))

    # Trying state-vector simulations
    qsimulator = QSimulator(backend = FakeQuitoV2())
    SV_count = qsimulator.State_Vector_Simulator(circuits = circuits_w_no_meas)
    op = Pauli('XIIII')
    print(SV_count.expectation_value(op))

    # the following makes me feel that we could leave this part just as a function in main ()
    experiment,noisedataframe = tomography_step( circuits = circuits_w_no_meas , pair_samples = 1, single_samples = 1, depths = [2,4,8,16],
                                                backend=backend_configuration.backend,initial_layout=backend_configuration.initial_layout, shots = 1000)
    #Noise_tomography.tomography_step(pair_samples = 1, single_samples = 1, depths = [2,4,8,16], circuit = circuits_w_no_meas)

    print("noise coefficients:", list(noisedataframe.noisemodels.values())[0].coeffs)
    print("spam coefficients:", list(zip(noisedataframe.spam.keys(), noisedataframe.spam.values())))

    tomo_time = time.time()
    print("time for tomo = ", tomo_time - start_time)

    ## ADD a check for layer decomposition


    #The exponential decay shows that the channel was properly diagonalized
    layer = experiment.analysis.get_layer_data(0)
    #layer.graph((0,),(1,),(0,1)) #input qubits and qubit links as tuples#The exponential decay shows that the channel was properly diagonalized
    layer = experiment.analysis.get_layer_data(0)
    #layer.graph((0,),(1,),(0,1)) #input qubits and qubit links as tuples
    print("here done with tomo")

    expectations = ["ZZZZX"]#["ZIIIIIIIIIIIIIII"]#,"IIIIX","IZIII","IIZII","IIIZI","ZIIII"] 


    circ_res = PER_step(experiment= experiment, expectations=expectations, samples=10, noise_strengths=[0,0.25,0.5],circuits = circuits_w_no_meas,
                     backend=backend_configuration.backend,initial_layout=backend_configuration.initial_layout, shots = 1000)
    #op = Pauli('ZIIIIIIIIIIIIIII')
    print(" Error Mitigated value foris", circ_res[0].get_result("ZZZZX").expectation)


    PER_time = time.time()
    print("time for tomo = ", -tomo_time + PER_time)

    return 

if __name__ == "__main__":
    main()


## TODO
# Add the state-vector, QASM and noisy backend code for comparision
# Make this code object oriented 
# Make the code cleaner 
# Running the Guadlupe on msi cluster 
# Update all the code here and then uload the files on cluster 

# think if we can add bond pairs for the QuantumSystem
# and can we loop over the bonds pairs to put the gates instead of typing everything again and again
# figure out how to add the measurements without making it spagetti 

# TODO 
# 1. MUST: Change circit for bonds list and check the circuit and uncomment the transpile function
# 2. Change num_qubit in hamittonian script to generalize.
# 1. UPDATE THIS SCRIPT FROM THE CLUSTER SCRIPT WITH SV AND QASM/NOISYBACKEND RESULTS FOR COMPARSION 
# 2. IMPORT JSON FILES FOR PAULI STRINGS
# 3. TRY AND EXPECT ROUTINE TO SEE IF PICKLE FOR TOMO EXIST THEN USE THAT OR RUN TOMOGRAPHY STEP 

