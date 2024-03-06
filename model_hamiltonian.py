# Essential imports for the calculations
from qiskit import QuantumCircuit, Aer, transpile, ClassicalRegister
from qiskit.visualization import plot_gate_map, plot_error_map, plot_histogram
from qiskit.providers.fake_provider import FakeQuitoV2
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit.providers.fake_provider import FakeCasablancaV2
from qiskit.quantum_info.operators.symplectic import Pauli
from matplotlib import pyplot as plt
import scipy
import sys
import numpy as np
import graphviz
import time 
from qiskit.quantum_info import SparsePauliOp 
from circuit_builder import CircuitBuilder
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
                paulis_str.append(''.join(list_s))
    elif num == 1:
        for pauli in [ham_pauli]:
            for i in range(num_qubits):
                list_s = list(s)
                list_s.insert(i, pauli)
                paulis_str.append(''.join(list_s))
    return paulis_str


def Hamiltonian_MFIM(bonds = []):
    """To build the Mixed-Field Ising model hamiltonian with the native qubit connectivitiy of the backend"""
    paulis_ZZ = ham_str_creation(num_qubits= 16,ham_pauli = "Z", bonds =bonds,num = 2)
    #print(paulis_ZZ)
    paulis_Z = ham_str_creation(num_qubits= 16, ham_pauli = "Z", bonds =bonds,num = 1)
    #print(paulis_Z)
    paulis_X = ham_str_creation(num_qubits= 16, ham_pauli = "X", bonds = bonds,num = 1)
    ham_ZZ = ["".join(reversed([p for p in pauli])) for pauli in paulis_ZZ]

    hamiltonian  = SparsePauliOp(ham_ZZ, coeffs = -1.0)+SparsePauliOp(paulis_Z, coeffs= 0.5)+SparsePauliOp(paulis_X, coeffs = -1)
    #print(hamiltonian)
    return hamiltonian 



def circuit_optimized_parameters(geometry = "FakeQuitoV2"): 
    """ this function consist of the optimized parameter for minimum energy via SV VQE optimization"""

    if geometry == "FakeQuitoV2":
        #initial_layout = [0, 1, 2, 3, 4]    
        # VQE solution for 1 layer HVA------- hardcoded here but are originally derived from optimizing the VQE solution. (Need to check)
        theta_Z_L_1 = [-1.0903836560221376]
        theta_X_L_1 = [1.5707963013100128]
        theta_ZZ_L_1 = [-1.290063556534689e-08]

            # VQE solution for 2 layer HVA for 4 qubit chain
            #theta_Z_L_2 = [-0.9253781962387742, 0.05297769164990435]
            #theta_X_L_2 = [1.1782568203539736, 0.44552055156550735]
            #theta_ZZ_L_2 = [0.2425000962970552, -0.10748314808466695]
    elif geometry == "FakeCasablancaV2":
            # Casablanca geometry
            # initial_layout = [0, 1, 2, 3, 4, 5, 6]
            # VQE solution for 1 layer HVA for Casablanca geometry
        theta_Z_L_1 = [-1.114862237442442]
        theta_X_L_1 = [1.5707966423051756]
        theta_ZZ_L_1 = [6.874680103745465e-07]

            # VQE solution for 2 layer HVA for Casablanca geometry
            #theta_Z_L_2 = [-1.0493592817846746, 0.07760329617674103]
            #theta_X_L_2 = [1.2057488386027533, 0.34794432057731883]
            #theta_ZZ_L_2 = [0.218276186042823, -0.16232253800006316]
    elif geometry == "FakeGuadalupeV2":
        # initial_layout = range(16)
        # VQE solution for 1 layer HVA for Quadalupe geometry 
        theta_Z_L_1 = -1.16677864
        theta_X_L_1 = 1.57079632
        theta_ZZ_L_1 = 4.90858079e-09
    else: 
        print("Geometry not supported so far")

    return [theta_Z_L_1, theta_X_L_1, theta_ZZ_L_1]