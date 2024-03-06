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
from model_hamiltonian import *



def ham_parameterized_circuit_sv(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2",bonds = []): #num_layers = 1,
    Vcircuit= CircuitBuilder(params, backend , initial_layout , geometry)
    circ_w_no_meas = [Vcircuit.makevqeCircuit(measure = False)]
    #print(circ_w_no_meas[0].draw())
    
    state = Aer.get_backend('statevector_simulator').run(circ_w_no_meas).result().get_statevector()
    # we will incorporate the pauli strings in qiskit way, i.e., little endian and will perform SV calns also in same way
    res = state.expectation_value(Hamiltonian_MFIM(bonds = bonds))
    res = np.real(res)
    # print(params, res)
    return res

def optimizer(init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], geometry = "FakeQuitoV2", bonds = []):
    res = scipy.optimize.minimize(ham_parameterized_circuit_sv, init_params, args = ( backend , initial_layout , geometry, bonds),method ="BFGS")
    return res

def VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], geometry = "FakeGuadalupeV2", bonds = []):
    res_vqe_sv = optimizer(init_params,backend,initial_layout,geometry, bonds)
    if res_vqe_sv.success:
        print('Optimization was successful!')
        optimum_params = res_vqe_sv.x
        lowest_fun = res_vqe_sv.fun
        print("optimized parameters are ", optimum_params," and lowest energy is  %f." %(lowest_fun))  
    else:
        print(res_vqe_sv.success)    
    
    return optimum_params, lowest_fun


def main():
    bonds_Guad = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 9), (8, 11), (11, 14), (14, 13),
                   (13, 12), (12, 15), (1, 4), (4, 7), (7, 6), (7, 10), (10, 12)]

    bonds_Quito = [[0, 1],[1, 2],[1, 3],[3, 4]]


    opt_params_Quito =  circuit_optimized_parameters("FakeQuitoV2")
    opt_params_Guad = circuit_optimized_parameters("FakeGuadalupeV2")

    init_params =  np.random.uniform(-np.pi/3, np.pi/3, 3)
    print(ham_parameterized_circuit_sv(params=opt_params_Guad,backend = FakeGuadalupeV2(), 
                                       initial_layout = [i for i in range(16)], geometry = "FakeGuadalupeV2", bonds = bonds_Guad)) #"FakeQuitoV2"))
    VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], geometry = "FakeGuadalupeV2", bonds=bonds_Guad)

    return

if __name__ == "__main__":
    main()

