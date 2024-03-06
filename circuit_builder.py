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



class Quantum_system:
    def __init__(self, backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2"):
        self.backend = backend
        #print("here backend",self.backend)
        self.initial_layout = initial_layout # could be different than all qubit connectivity as used in this project
        self.geometry = geometry
        
class CircuitBuilder(Quantum_system):
    def __init__(self,params,backend , initial_layout , geometry):
        super().__init__(backend,initial_layout,geometry)
        #print("backend here is ", self.backend)
        
        #self.backend = backend
        #print("backend",self.backend)
        #self.initial_layout = initial_layout
        #print("self.initlayout", self.initial_layout)
        #self.geometry = geometry
        #print("geometry", self.geometry)
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
    
    def vqeLayer_FakeGuadalupeV2(self,params):
        """ VQE layer for the FakeGuadalupeV2() geometry using all qubits and native connectivity"""
        theta_Z = params[0] # make bonds a list of bond list 
        theta_X = params[1]
        theta_ZZ = params[2]
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
            #print("i = ",i)
            if self.geometry == "FakeCasablancaV2":
                vqeCircuit.h(range(self.num_qubits)) # initialize in the |+> state
                vqeCircuit.barrier()
                vqeL = self.vqeLayer_Casablanca(self.params[3*i:3*(i+1)]) # self.theta_ZZ_L_1[i], self.theta_Z_L_1[i], self.theta_X_L_1[i])
            elif self.geometry == "FakeQuitoV2":
                #print("here")
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
    