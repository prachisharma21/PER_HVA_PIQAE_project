# Essential imports for the calculations
from qiskit import QuantumCircuit, Aer, transpile, ClassicalRegister
from qiskit.visualization import plot_gate_map, plot_error_map, plot_histogram
from qiskit.providers.fake_provider import FakeQuitoV2
from qiskit.providers.fake_provider import FakeGuadalupeV2
#from qiskit.providers.fake_provider import FakeCasablancaV2
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

def quantum_input(backend = FakeQuitoV2()):
    """ function is to get the basic essentials inputs for the quantum device from the user """    
    backend = backend 
    num_qubits = backend.num_qubits
    return backend, num_qubits

backend, num_qubits = quantum_input(backend = FakeQuitoV2()) #FakeGuadalupeV2())

def model_input(geometry = "FakeQuito", J = -1, hx = -1, hz = 0.5, backend_bond_pairs =  [[0, 1],[1, 2],[1, 3],[3, 4]]):
    """ this function gets the input for the problem hamiltonian: Mixed Ising field model and the bond connectivity of the backend for HVA"""
    J = J
    hx = hx 
    hz = hz
    bonds = backend_bond_pairs
    if geometry == "FakeQuito":
            initial_layout = [0, 1, 2, 3, 4]    
            # VQE solution for 1 layer HVA 
            theta_Z_L_1 = [-1.0903836560221376]
            theta_X_L_1 = [1.5707963013100128]
            theta_ZZ_L_1 = [-1.290063556534689e-08]

            # VQE solution for 2 layer HVA for 4 qubit chain
            #theta_Z_L_2 = [-0.9253781962387742, 0.05297769164990435]
            #theta_X_L_2 = [1.1782568203539736, 0.44552055156550735]
            #theta_ZZ_L_2 = [0.2425000962970552, -0.10748314808466695]
    elif geometry == "Casablanca":
            # Casablanca geometry
            initial_layout = [0, 1, 2, 3, 4, 5, 6]
            # VQE solution for 1 layer HVA for Casablanca geometry
            theta_Z_L_1 = [-1.114862237442442]
            theta_X_L_1 = [1.5707966423051756]
            theta_ZZ_L_1 = [6.874680103745465e-07]

            # VQE solution for 2 layer HVA for Casablanca geometry
            #theta_Z_L_2 = [-1.0493592817846746, 0.07760329617674103]
            #theta_X_L_2 = [1.2057488386027533, 0.34794432057731883]
            #theta_ZZ_L_2 = [0.218276186042823, -0.16232253800006316]
    elif geometry == "FakeGuadalupe":
            initial_layout = range(16)
            # VQE solution for 1 layer HVA for Quadalupe geometry 
            theta_Z_L_1 = [-1.16677864]
            theta_X_L_1 = [1.57079632]
            theta_ZZ_L_1 = [4.90858079e-09]
    else: 
            print("Geometry not supported so far, i.e., no VQE solution provided.")
    return geometry, initial_layout, theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1

geometry,initial_layout, theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1 = model_input(geometry = "FakeQuito", J = -1, hx = -1, hz = 0.5, backend_bond_pairs =  [[0, 1],[1, 2],[1, 3],[3, 4]])  
print(initial_layout, theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1) 

def vqeLayer_FakeQuito(theta_ZZ, theta_Z, theta_X, num_qubits = FakeQuitoV2.num_qubits):
    """ VQE layer for the FakeQuito geometry using all qubits and native connectivity"""
    vqeLayer = QuantumCircuit(num_qubits)
    # Choosen bond pairs according to the native qubit connectivity of the backend
    bonds_1 = [[0, 1], [3, 4]]
    bonds_2 = [[1, 2]]
    bonds_3 = [[1, 3]]
    # the RZ and RZ terms for the field terms of the hamiltonian. 
    #Applied first to get the sequence of layers for PER later to come out correctly, i.e., single qubit gates first followed by clifford gates. 
    vqeLayer.rz(theta_Z, range(num_qubits))
    vqeLayer.rx(theta_X, range(num_qubits))
    
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

def vqeLayer_FakeGuadalupeV2(theta_ZZ, theta_Z, theta_X, num_qubits = FakeGuadalupeV2.num_qubits):
    """ VQE layer for the FakeGuadalupeV2() geometry using all qubits and native connectivity"""
    vqeLayer = QuantumCircuit(num_qubits)
    # Choosen bond pairs according to the native qubit connectivity of the backend
    bonds_1 = [(0, 1), (2, 3), (4, 7), (10, 12)]
    bonds_2 = [[1, 2], [3, 5],[7,6],[8,9],[12,13]] 
    bonds_3 = [[1, 4], [7,10],[12,15]] # ,[8,11]
    bonds_4 = [(5, 8) ,(11, 14)]
    bonds_5 = [[8,11]]

    vqeLayer.rz(theta_Z, range(num_qubits))
    vqeLayer.rx(theta_X, range(num_qubits))
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

def makevqeCircuit_no_meas(theta_ZZ, theta_Z, theta_X, initial_layout = initial_layout, geometry = geometry):
    num_qubits = len(initial_layout)
    vqeCircuit = QuantumCircuit(num_qubits)
    for i in range(len(theta_ZZ)):
        if geometry == "Casablanca":
            vqeCircuit.h(range(num_qubits)) # initialize in the |+> state
            vqeCircuit.barrier()
            vqeL = vqeLayer_Casablanca(theta_ZZ[i], theta_Z[i], theta_X[i])
        elif geometry == "FakeQuito":
            vqeCircuit.h(range(num_qubits)) # initialize in the |+> state
            vqeCircuit.barrier()
            vqeL = vqeLayer_FakeQuito(theta_ZZ[i], theta_Z[i], theta_X[i], num_qubits)
        elif geometry == "FakeGuadalupe":
            vqeCircuit.h(range(num_qubits)) # initialize in the |+> state
            vqeCircuit.barrier()
            vqeL = vqeLayer_FakeGuadalupeV2(theta_ZZ[i], theta_Z[i], theta_X[i], num_qubits)
                     
        vqeCircuit = vqeCircuit.compose(vqeL)
        vqeCircuit.barrier()
       
    transpiled = vqeCircuit 
    #transpiled = transpile(vqeCircuit, backend, initial_layout = initial_layout)
    return transpiled


circuits_vqe_no_meas = [makevqeCircuit_no_meas(theta_ZZ_L_1, theta_Z_L_1, theta_X_L_1, initial_layout)]

#print(circuits_vqe_no_meas[0])


## PER functions and steps 
shots = 1000


def executor(circuits):
    return backend.run(circuits, shots = shots).result().get_counts()


def tomography_step(pair_samples = 1, single_samples = 1, depths = [2,4,8,16], circuit = circuits_vqe_no_meas):
    print(type(circuits_vqe_no_meas))
    experiment = tomography(circuits = circuits_vqe_no_meas, inst_map = initial_layout, backend = backend) # add initial layout as a parameter here
    experiment.generate(samples = pair_samples, single_samples = single_samples, depths = depths)
    #run the experiment
    experiment.run(executor)
    noisedataframe = experiment.analyze()
    return experiment, noisedataframe

experiment,noisedataframe = tomography_step(pair_samples = 1, single_samples = 1, depths = [2,4,8,16], circuit = circuits_vqe_no_meas[0])

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

def PER_step(expt = experiment, expectation = expectations, samples = 10, noise_strengths = [0, 0.25, 0.5]):
    perexp = experiment.create_per_experiment(circuits_vqe_no_meas)
    perexp.generate(expectations = expectations, samples = samples, noise_strengths = noise_strengths)
    print(perexp.meas_bases)
    perexp.run(executor)
    circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
    #and apply vZNE
    return circuit_results

circ_res = PER_step(experiment, expectation=expectations, samples=10, noise_strengths=[0,0.25,0.5])
op = Pauli('ZIIIIIIIIIIIIIII')
print(" Error Mitigated value foris", circ_res[0].get_result("ZZZZX").expectation)


PER_time = time.time()
print("time for tomo = ", -tomo_time + PER_time)


## TODO
# Add the state-vector, QASM and noisy backend code for comparision
# Make this code object oriented 
# Make the code cleaner 
# Running the Guadlupe on msi cluster 