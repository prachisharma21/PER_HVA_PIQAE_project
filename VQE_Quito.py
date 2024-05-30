# Essential imports for the calculations
from qiskit.providers.fake_provider import FakeQuitoV2, FakeGuadalupeV2
import scipy
import numpy as np

from circuit_builder import CircuitBuilder
from model_hamiltonian import Hamiltonian_MFIM,circuit_optimized_parameters
from helper import QSimulator


# Here VQE is done with SV simulator 
def VQE_parameterized_circuit_sv(params =[],backend = FakeQuitoV2(),initial_layout = [0, 1, 2, 3, 4],bonds = [], num_layers = 1):
    """
    Builds a parameterized circuits to perform the VQE step using HVA ansatz
    """
    Vcircuit= CircuitBuilder(params, backend, initial_layout, nlayers = num_layers)
    circ_w_no_meas = [Vcircuit.makevqeCircuit(measure = False)]
    qsim = QSimulator(backend)
    SV_counts = qsim.State_Vector_Simulator(circ_w_no_meas) 
    res = SV_counts.expectation_value(Hamiltonian_MFIM(bonds = bonds))
    res = np.real(res)
    return res

def optimizer(init_params,backend = FakeQuitoV2(), initial_layout = [0, 1, 2, 3, 4], bonds = []):
    """
    Classical optimization step of VQE
    """
    res = scipy.optimize.minimize(VQE_parameterized_circuit_sv, init_params, args = ( backend, initial_layout, bonds),method ="BFGS")
    return res

def VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], bonds = []):
    res_vqe_sv = optimizer(init_params,backend,initial_layout, bonds)
    if res_vqe_sv.success:
        print('Optimization was successful!')
        optimum_params = res_vqe_sv.x
        lowest_fun = res_vqe_sv.fun
        print("optimized parameters are ", optimum_params," and lowest energy is  %f." %(lowest_fun))  
    else:
        print(res_vqe_sv.success)    
    
    return optimum_params, lowest_fun


def main():
    # native connectivity of the Guadalupe backend
    bonds_Guad = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 9), (8, 11), (11, 14), (14, 13),
                   (13, 12), (12, 15), (1, 4), (4, 7), (7, 6), (7, 10), (10, 12)]
    # native connectivity of Quito backend
    bonds_Quito = [[0, 1],[1, 2],[1, 3],[3, 4]]
    # there are three parameters for each layer in this VQE ansatz 
    num_params = 3 
    # number of layers for HVA ansatz 
    num_layers = 1 
    # optimized parameters for the VQE step as used in the paper
    opt_params_Quito =  circuit_optimized_parameters("FakeQuitoV2")
    opt_params_Guad = circuit_optimized_parameters("FakeGuadalupeV2")

    # Chosing random parameters to perform VQE
    init_params =  np.random.uniform(-np.pi/3, np.pi/3, num_layers*num_params)
    print(VQE_parameterized_circuit_sv(params = init_params,backend = FakeGuadalupeV2(), 
                                       initial_layout = [i for i in range(16)], bonds = bonds_Guad, num_layers=num_layers)) 
    
    VQE_optimization_Step(init_params,backend = FakeGuadalupeV2(), initial_layout = [i for i in range(16)], bonds=bonds_Guad)

    return

if __name__ == "__main__":
    main()

