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

plt.style.use("ggplot")
#print(np.pi)
start_time = time.time()

class Quantum_system:
    def __init__(self, backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"):
        self.backend = backend
        self.initial_layout = initial_layout # could be different than all qubit connectivity as used in this project
        self.geometry = geometry
        self.theta_Z_L_1, self.theta_X_L_1, self.theta_ZZ_L_1 = self.circuit_parameters()

    def circuit_parameters(self): 
        """ this function gets the input for the problem hamiltonian: Mixed Ising field model and the bond connectivity of the backend for HVA"""

        if self.geometry == "FakeQuitoV2":
            #initial_layout = [0, 1, 2, 3, 4]    
            # VQE solution for 1 layer HVA------- hardcoded here but are originally derived from optimizing the VQE solution. (Need to check)
            theta_Z_L_1 = [-1.0903836560221376]
            theta_X_L_1 = [1.5707963013100128]
            theta_ZZ_L_1 = [-1.290063556534689e-08]

            # VQE solution for 2 layer HVA for 4 qubit chain
            #theta_Z_L_2 = [-0.9253781962387742, 0.05297769164990435]
            #theta_X_L_2 = [1.1782568203539736, 0.44552055156550735]
            #theta_ZZ_L_2 = [0.2425000962970552, -0.10748314808466695]
        elif self.geometry == "FakeCasablancaV2":
            # Casablanca geometry
            # initial_layout = [0, 1, 2, 3, 4, 5, 6]
            # VQE solution for 1 layer HVA for Casablanca geometry
            self.theta_Z_L_1 = [-1.114862237442442]
            self.theta_X_L_1 = [1.5707966423051756]
            self.theta_ZZ_L_1 = [6.874680103745465e-07]

            # VQE solution for 2 layer HVA for Casablanca geometry
            #theta_Z_L_2 = [-1.0493592817846746, 0.07760329617674103]
            #theta_X_L_2 = [1.2057488386027533, 0.34794432057731883]
            #theta_ZZ_L_2 = [0.218276186042823, -0.16232253800006316]
        elif self.geometry == "FakeGuadalupeV2":
            # initial_layout = range(16)
            # VQE solution for 1 layer HVA for Quadalupe geometry 
            self.theta_Z_L_1 = [-1.16677864]
            self.theta_X_L_1 = [1.57079632]
            self.theta_ZZ_L_1 = [4.90858079e-09]
        else: 
            print("Geometry not supported so far")

        return theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1



def model_input( J = -1, hx = -1, hz = 0.5):
    """ this function gets the input for the problem hamiltonian: Mixed Ising field model for HVA"""
    J = J
    hx = hx 
    hz = hz
    return J, hx, hz

class CircuitBuilder(Quantum_system):
    def __init__(self,backend , initial_layout , geometry):
        super().__init__(backend,initial_layout,geometry)
        print("backend here is ", self.backend)
        self.num_qubits = len(self.initial_layout)

    def vqeLayer_FakeQuito(self,theta_ZZ, theta_Z, theta_X):
        """ VQE layer for the FakeQuito geometry using all qubits and native connectivity"""
        vqeLayer = QuantumCircuit(self.num_qubits)
        # Choosen bond pairs according to the native qubit connectivity of the backend
        bonds_1 = [[0, 1], [3, 4]]  # these bond pairs should be accessible as well-- they will be needed for hamiltonian expectation for eg. 
        bonds_2 = [[1, 2]] # could be added to the main QuantumSystem class as they are dependent on the backend geometry
        bonds_3 = [[1, 3]]
        # the RZ and RZ terms for the field terms of the hamiltonian. 
        #Applied first to get the sequence of layers for PER later to come out correctly, i.e., single qubit gates first followed by clifford gates. 
        
        vqeLayer.rz(theta_Z, range(self.num_qubits))
        vqeLayer.rx(theta_X, range(self.num_qubits))
    
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        vqeLayer.rz(theta_ZZ, [bonds_1[i][1] for i in range(len(bonds_1))])
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))

        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        vqeLayer.rz(theta_ZZ, [bonds_2[i][1] for i in range(len(bonds_2))])
        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))

        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        vqeLayer.rz(theta_ZZ, [bonds_3[i][1] for i in range(len(bonds_3))])
        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
   
        #vqeLayer.barrier()

        return vqeLayer
    
    def vqeLayer_FakeGuadalupeV2(self,theta_ZZ, theta_Z, theta_X):
        """ VQE layer for the FakeGuadalupeV2() geometry using all qubits and native connectivity"""
        vqeLayer = QuantumCircuit(self.num_qubits)
        # Choosen bond pairs according to the native qubit connectivity of the backend
        bonds_1 = [(0, 1), (2, 3), (4, 7), (10, 12)]
        bonds_2 = [[1, 2], [3, 5],[7,6],[8,9],[12,13]] 
        bonds_3 = [[1, 4], [7,10],[12,15]] # ,[8,11]
        bonds_4 = [(5, 8) ,(11, 14)]
        bonds_5 = [[8,11]]

        vqeLayer.rz(theta_Z, range(self.num_qubits))
        vqeLayer.rx(theta_X, range(self.num_qubits))
        #vqeLayer.barrier()
    
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        vqeLayer.rz(theta_ZZ, [bonds_1[i][1] for i in range(len(bonds_1))])
        vqeLayer.cx(*zip(*[bonds_1[i] for i in range(len(bonds_1))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        vqeLayer.rz(theta_ZZ, [bonds_2[i][1] for i in range(len(bonds_2))])
        vqeLayer.cx(*zip(*[bonds_2[i] for i in range(len(bonds_2))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        vqeLayer.rz(theta_ZZ, [bonds_3[i][1] for i in range(len(bonds_3))])
        vqeLayer.cx(*zip(*[bonds_3[i] for i in range(len(bonds_3))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_4[i] for i in range(len(bonds_4))]))
        vqeLayer.rz(theta_ZZ, [bonds_4[i][1] for i in range(len(bonds_4))])
        vqeLayer.cx(*zip(*[bonds_4[i] for i in range(len(bonds_4))]))
        #vqeLayer.barrier()

        vqeLayer.cx(*zip(*[bonds_5[i] for i in range(len(bonds_5))]))
        vqeLayer.rz(theta_ZZ, [bonds_5[i][1] for i in range(len(bonds_5))])
        vqeLayer.cx(*zip(*[bonds_5[i] for i in range(len(bonds_5))]))
    
        #vqeLayer.barrier()
        return vqeLayer
    
    # Having intial_layout is better---as then you can choose not to use all qubits. 
    def makevqeCircuit(self, measure = False, meas_basis = "Z"): # NEED to figure out how to input measure and basis here 
        
        vqeCircuit = QuantumCircuit(self.num_qubits)
        for i in range(len(self.theta_ZZ_L_1)):
            if self.geometry == "FakeCasablancaV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_Casablanca(self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeQuitoV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_FakeQuito(self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeGuadalupeV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_FakeGuadalupeV2(self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            vqeCircuit = vqeCircuit.compose(vqeL)
            vqeCircuit.barrier()
               
        if measure == True:
            if meas_basis == "Z":
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            elif meas_basis =="X":
                vqeCircuit.h(range(self.num_qubits))
                vqeCircuit.measure_all()
                transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
            else: 
                print("Measurement not defined")    # Y-measurements can be added
        else: 
            transpiled = transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
        return transpiled


#circuits_vqe_no_meas = [makevqeCircuit_no_meas(theta_ZZ_L_1, theta_Z_L_1, theta_X_L_1, initial_layout)]

#print(circuits_vqe_no_meas[0])


## PER functions and steps 

class Noise_tomography(Quantum_system):
    def __init__(self, circuit, pair_samples, single_samples, depth):
        self.circuit = circuit
        self.pair_samples = pair_samples
        self.single_samples = single_samples
        self.depth = depth
        super.__init__()

    def tomography_step(self):#, pair_samples = 1,single_samples =1, depth = [2,4,8,16]):
        experiment = tomography(circuits = self.circuits, inst_map = self.initial_layout, backend = self.backend) # add initial layout as a parameter here
        # to generate the circuits of different depths
        experiment.generate(samples = self.pair_samples, single_samples = self.single_samples, depths = self.depth)
        #run the experiment
        experiment.run(executor)
        noisedataframe = experiment.analyze()
        return experiment, noisedataframe
    
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


def main():
  
    backend_configuration = Quantum_system(backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")  #FakeGuadalupeV2())
    circuits = CircuitBuilder(backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")
    #print("Backend =", backend)
    #Circuits sent to tomography has to be a list
    circuits_w_no_meas = [circuits.makevqeCircuit(measure = False)]
    print(circuits_w_no_meas[0].draw())
    #print(type(circuits_w_no_meas))
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


