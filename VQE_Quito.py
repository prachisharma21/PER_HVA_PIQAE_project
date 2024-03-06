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
from qiskit.quantum_info import SparsePauliOp 

# the classes CircuitBuilder and QuantumSystem are in PER script and can be used here 
# except we need to change the parameters
# Quantum_system
class CircuitBuilder():
    def __init__(self,params,backend , initial_layout , geometry):
        #super().__init__(backend,initial_layout,geometry,params)
        #print("backend here is ", self.backend)
        
        self.backend = backend
        print("backend",self.backend)
        self.initial_layout = initial_layout
        print("self.initlayout", self.initial_layout)
        self.geometry = geometry
        print("geometry", self.geometry)
        self.params = params
        self.num_qubits = len(self.initial_layout)
       

    def vqeLayer_FakeQuito(self,params):#theta_ZZ, theta_Z, theta_X): #params has to be a list a list for each layer
        """ VQE layer for the FakeQuito geometry using all qubits and native connectivity"""
        theta_Z = params[0] # make bonds a list of bond list 
        theta_X = params[1]
        theta_ZZ = params[2]
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
        for i in range(int(len(self.params)/3)): #len params list /number of parameters in the circuit => 6/3-1 = 1 and range(1)=>[0,1]
            print("i = ",i)
            if self.geometry == "FakeCasablancaV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_Casablanca(self.params[3*i:3*(i+1)]) # self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeQuitoV2":
                print("here")
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_FakeQuito(self.params[3*i:3*(i+1)]) #self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeGuadalupeV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_FakeGuadalupeV2(self.params[3*i:3*(i+1)])#self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
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
            transpiled = vqeCircuit # transpile(vqeCircuit, self.backend, initial_layout = self.initial_layout)
        return transpiled
    



def ham_str_creation(num_qubits = 5,ham_pauli = "Z", bonds =[], num = 2):
    paulis_str = []
    s = "I"*(num_qubits - num) 
    if num == 2:
        for pauli in [ham_pauli]:
            for bond in bonds: 
            #print(bond)
                list_s = list(s)
                list_s.insert(bond[0], pauli)
                list_s.insert(bond[1], pauli)
            # if else below is only if you are not using SparsePauliOp 
            #if pauli == "X":
            #    pauli_string="1.0*" + ''.join(list_s)
            #elif pauli == "Y":
            #    pauli_string="1.0*" + ''.join(list_s)
            #elif pauli == "Z":
            #    pauli_string="-1.0*" + ''.join(list_s)
                paulis_str.append(''.join(list_s))
                #print(paulis_str)
            #ham.append(pauli_string)
    elif num == 1:
        for pauli in [ham_pauli]:
            for i in range(num_qubits):
                list_s = list(s)
                list_s.insert(i, pauli)
                paulis_str.append(''.join(list_s))
    return paulis_str
        
def Hamiltonian_MFIM():
    paulis_ZZ = ham_str_creation(5,ham_pauli = "Z", bonds = [[0, 1],[1, 2],[1, 3],[3, 4]],num = 2)
    #print(paulis_ZZ)
    paulis_Z = ham_str_creation(5, ham_pauli = "Z", bonds = [[0, 1],[1, 2],[1, 3],[3, 4]],num = 1)
    #print(paulis_Z)
    paulis_X = ham_str_creation(5, ham_pauli = "X", bonds = [[0, 1],[1, 2],[1, 3],[3, 4]],num = 1)
    ham_ZZ = ["".join(reversed([p for p in pauli])) for pauli in paulis_ZZ]

    hamiltonian  = SparsePauliOp(ham_ZZ, coeffs = -1.0)+SparsePauliOp(paulis_Z, coeffs= 0.5)+SparsePauliOp(paulis_X, coeffs = -1)

    return hamiltonian 

#print(Hamiltonian_MFIM())

    
bonds = [[0, 1],[1, 2],[1, 3],[3, 4]] # self,backend , initial_layout , geometry,params
def ham_pqc_sv(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"): #num_layers = 1,
    Vcircuit= CircuitBuilder(params, backend , initial_layout , geometry)
    circ_w_no_meas = [Vcircuit.makevqeCircuit(measure = False)]
    #print(circ_w_no_meas[0].draw())
    
    state = Aer.get_backend('statevector_simulator').run(circ_w_no_meas).result().get_statevector()
    # we will incorporate the pauli strings in qiskit way, i.e., little endian and will perform SV calns also in same way
    #state_tf = little_to_big_endian(state, num_qubits)
    #if bonds_all == bonds_kagome_5:     
    #    ham_matrix = np.array(map_hamiltonian(ham_kagome_5))
    #else:
    #    print("Hamiltonian matrix not defined.")
    #res = np.vdot(state_tf, ham_matrix.dot(state_tf))
    res = state.expectation_value(Hamiltonian_MFIM())
    res = np.real(res)
    print(res)
    return res
#print("length check = ",int(len([1,2,3,4,5,6])/3))
#theta_Z_L_1 = [-1.0903836560221376]
#theta_X_L_1 = [1.5707963013100128]
#theta_ZZ_L_1 = [-1.290063556534689e-08]
init_params =  np.random.uniform(-np.pi/7, np.pi/7, 3)
# [np.pi,np.pi/2,np.pi/4]#[-1,1.2,0.1]#[-1.0903836560221376,1.5707963013100128,-1.290063556534689e-08]
#[np.pi,np.pi/2,np.pi/4]
print(ham_pqc_sv(params=init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"))
import scipy
#params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"
def optimizer(init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"):
    res = scipy.optimize.minimize(ham_pqc_sv, init_params, args = ( backend , initial_layout , geometry),method ="BFGS")
    return res

res_vqe_sv=optimizer(init_params)
print(res_vqe_sv)


    #backend_configuration = Quantum_system(backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")  #FakeGuadalupeV2())
    #circuits = CircuitBuilder(backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2")