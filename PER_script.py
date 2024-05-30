# Essential imports for the calculations
from qiskit.providers.fake_provider import FakeQuitoV2, FakeGuadalupeV2
from qiskit.quantum_info.operators.symplectic import Pauli
from matplotlib import pyplot as plt

import sys
import argparse

sys.path.append("/home/prachi/Documents/Code_PER/AutomatedPERTools-BEN_OG/AutomatedPERTools-main/pauli_lindblad_per/")
from tomography.experiment import SparsePauliTomographyExperiment as tomography
#from primitives.pauli import QiskitPauli

from helper import *
from circuit_builder import CircuitBuilder
from model_hamiltonian import circuit_optimized_parameters
plt.style.use("ggplot")
    

def tomography_step( circuits = None, pair_samples = 1,single_samples =1, depths = [2,4,8,16]
                    , backend = FakeQuitoV2(), initial_layout = [],shots = 1000):
    """
    Performs the noise tomography steps for the circuits. 

    Inputs:
    circuits: list of circuits
    pair_samples: number of samples for pair fidelity measurements
    single_samples: number of samples for single fidelity measurements
    depths: list of circuits depths
    backend: fake backend chosen for the calculations
    initial_layout: list of number of qubits used  

    Outputs:
    --Experiment python object on which one performs the PER step
    --Obtained noisedataframe for analysis the noise model
    """
    experiment = tomography(circuits = circuits, inst_map = initial_layout, backend = backend) # add initial layout as a parameter here
    # to generate the circuits of different depths
    experiment.generate(samples = pair_samples, single_samples = single_samples, depths =depths)
    #run the experiment
    Quant_sys = Quantum_system(backend=backend)
    experiment.run(Quant_sys.executor)
    noisedataframe = experiment.analyze()
    return experiment, noisedataframe

def PER_step(experiment = None, expectations = [], samples = 10, noise_strengths = [0, 0.25, 0.5],
             circuits = None,backend = FakeQuitoV2()):
    """
    Performs the Probabilistic Error Reduction(PER) step. 

    Inputs:
    experiment: Tomography experiment python object
    expectations: list of observales which we desire to evaluate expectation of
    samples: number of PER circuits to be evaluated 
    noise_strengths: list of noise strengths
    circuits: list of circuits
    backend: fake backend chosen for the calculations
    
    Outputs:
    --circuit_result python object after performing PER step
    
    """
    perexp = experiment.create_per_experiment(circuits)
    perexp.generate(expectations = expectations, samples = samples, noise_strengths = noise_strengths)
    Quant_sys = Quantum_system(backend=backend)
    perexp.run(Quant_sys.executor)
    circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
    #and apply vZNE
    return circuit_results

def Create_PER_expt_pickle(circuits = None, pair_samples = 1,single_samples =1, depths = [2,4,8,16]
                    , backend = FakeQuitoV2(), initial_layout = [],shots = 1000):
    """
    Performs the noise tomography step and creates a perexpt object to further perform PER step. 

    Inputs:
    circuits: list of circuits
    pair_samples: number of samples for pair fidelity measurements
    single_samples: number of samples for single fidelity measurements
    depths: list of circuits depths
    backend: fake backend chosen for the calculations
    initial_layout: list of number of qubits used  

    """
    
    tomo_exp = tomography(circuits = circuits, inst_map = initial_layout, backend = backend) # add initial layout as a parameter here
    # to generate the circuits of different depths
    tomo_exp.generate(samples = pair_samples, single_samples = single_samples, depths =depths)
    #run the experiment
    Quant_sys = Quantum_system(backend=backend, initial_layout=initial_layout)
    tomo_exp.run(Quant_sys.executor)
    # created a noisedataframe which can provide information about the noise model
    noisedataframe = tomo_exp.analyze()
    # create a perexp object which will be used to generate PER circuits and perform PER step
    perexp = tomo_exp.create_per_experiment(circuits)
    # saved the perexp object as pickle to avoid running tomography step each time. 
    data_dump_pickle(output_data=perexp, output_file_name= "per_experiment.pickle")
    return 

def PER_step_with_pickle(perexpt = None, expectation = [], samples = 10, noise_strengths = [0, 0.25, 0.5], backend = FakeQuitoV2()):
    """
    Performs the Probabilistic Error Reduction(PER) step by loading the existing perexp pickle object to skip tomography step. 

    Inputs:
    perexpt: PER_experiment python object
    expectation: list of observales which we desire to evaluate expectation of
    samples: number of PER circuits to be evaluated 
    noise_strengths: list of noise strengths
    circuits: list of circuits
    backend: fake backend chosen for the calculations
    
    Outputs:
    --measurement basis for the list of observables passed in expectations
    --overhead while performing the PER step
    --circuit_result python object after performing PER step
    --perexpt object needed further analysis
    
    """
    perexp = perexpt # experiment.create_per_experiment(circuits_vqe_no_meas)
    perexp.generate(expectations = expectation, samples = samples, noise_strengths = noise_strengths)
    # print(perexp.meas_bases)
    Quant_sys = Quantum_system(backend=backend)
    perexp.run(Quant_sys.executor) # CHECK what backend goes here
    overhead = []
    for ns in noise_strengths:
        overhead.append(perexp.get_overhead(0,ns))
    
    circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
    #and apply vZNE
    return perexp.meas_bases,overhead,circuit_results,perexp

def PER_mitigated_expectation():
    """
    This function demonstrate the entire procedure of performing tomography to obtain a noise model 
    and then performing PER to obtain the mitigated expectation value of the observables. 
    Outputs: 
    --ideal expectation value obtained from state-vector simulations
    --error mitigated expectation value using PER 

    """
    # parameters for Quantum backend 
    choosen_backend = FakeQuitoV2()
    init_layout = [i for i in range(choosen_backend.num_qubits)]
    Quant_sys = Quantum_system(backend=choosen_backend, initial_layout=init_layout)
    geometry = Quant_sys.backend_geometry()

    # parameters for creating the VQE circuit 
    num_layers = 1
    opt_params_Quito =  circuit_optimized_parameters(geometry)
    circuits = CircuitBuilder(params = opt_params_Quito, backend =choosen_backend, initial_layout = init_layout, geometry =geometry , nlayers = num_layers)
    circuits_w_no_meas = [circuits.makevqeCircuit(measure = False)] #Circuits sent to tomography has to be a list
    print(circuits_w_no_meas[0].draw())
    
    # parameters for performing tomography
    depths_for_tomography = [2,4,8,16]
    shots = 1000

    # parameters for the PER step
    noise_strengths_for_PER = [0,0.25,0.5,1] 
    num_samples = 10

    # observable to be measured 
    expectations = ["XIIII"]     #["ZIIIIIIIIIIIIIII"]#,"IIIIX","IZIII","IIZII","IIIZI","ZIIII"] 

    # State-vector simulations to find ideal expectation value 
    qsimulator = QSimulator(backend = choosen_backend)
    SV_count = qsimulator.State_Vector_Simulator(circuits = circuits_w_no_meas)
    operator_to_meas  = Pauli(expectations[0])
    ideal_expectation = SV_count.expectation_value(operator_to_meas)

    # the following makes me feel that we could leave this part just as a function in main ()
    experiment,noisedataframe = tomography_step( circuits = circuits_w_no_meas , pair_samples = 1, single_samples = 1, depths = depths_for_tomography ,
                                                backend = choosen_backend, initial_layout = init_layout, shots = shots)

    #print("noise coefficients:", list(noisedataframe.noisemodels.values())[0].coeffs)
    #print("spam coefficients:", list(zip(noisedataframe.spam.keys(), noisedataframe.spam.values())))
    #The exponential decay shows that the channel was properly diagonalized
    #layer = experiment.analysis.get_layer_data(0)
    #layer.graph((0,),(1,),(0,1)) #input qubits and qubit links as tuples#The exponential decay shows that the channel was properly diagonalized
    #layer = experiment.analysis.get_layer_data(0)
    #layer.graph((0,),(1,),(0,1)) #input qubits and qubit links as tuples

    circ_res = PER_step(experiment= experiment, expectations=expectations, samples=num_samples, 
                        noise_strengths=noise_strengths_for_PER, circuits = circuits_w_no_meas,
                     backend = choosen_backend, initial_layout = init_layout, shots = shots)
    #op = Pauli('ZIIIIIIIIIIIIIII')
    mitigated_expectation_value = circ_res[0].get_result("XIIII").expectation

    print(f" Ideal expectation value for {expectations[0]} is {ideal_expectation}")
    print(f" Error Mitigated value for {expectations[0]} is {mitigated_expectation_value}")

    return ideal_expectation, mitigated_expectation_value



def main():

    # VQE ansatz: HVA layers        
    num_layers = 1

    ### parameters for Quantum backend 
    choosen_backend = FakeGuadalupeV2() 
    init_layout = [i for i in range(choosen_backend.num_qubits)]
    # initiate the Quantum_system class
    Quant_sys = Quantum_system(backend=choosen_backend, initial_layout=init_layout)
    geometry = Quant_sys.backend_geometry()

    # loading the optimized parameters for each backend geometry based on VQE minimization step
    # One could run the VQE.py to find the optimized parameters but used parameters in the papers are saved here for reproducibility purposes. 
    opt_params =  circuit_optimized_parameters(geometry)
    circuits = CircuitBuilder(params = opt_params, backend =choosen_backend, initial_layout = init_layout, nlayers = num_layers)

    # Circuits sent to tomography are a list
    circuits_w_no_meas = [circuits.makevqeCircuit(measure = False)]
    print("Circuits with one HVA layer ansatz is ", circuits_w_no_meas[0].draw())

    ### parameters for performing tomography
    # depths of the circuits while performing tomography step
    depths_for_tomography = [2,4,8,16]
    # number of shots to excecute each circuit
    shots = 1

    ###  parameters for the PER step
    # list of noise strengths for PER
    noise_strengths_for_PER = [0,0.25,0.5] 
    # number of PER circuits to be generated 
    num_samples = 1 

    # Try to load the PER object created after tomography step to avoid repeating time-comsuimg tomography step.   
    try:
        per_experiment = data_loader_pickle("per_experiment.pickle")
        print("Reading the existing per_experiment.pickle file")
    except:
        print("per_experiment.pickle file doesn't exist: Running the tomography step to create a per_expt object")
        Create_PER_expt_pickle(circuits = circuits_w_no_meas , pair_samples = 1, single_samples = 1, depths = depths_for_tomography ,
                                                backend = choosen_backend, initial_layout = init_layout, shots = shots)
        per_experiment = data_loader_pickle("per_experiment.pickle")


    parser = argparse.ArgumentParser(description="Inputs for pauli groups file name and select range of which elements to run")

    parser.add_argument("--Pauligroups", type=str, help="File name of  Pauli groups of commuting observables")
    parser.add_argument("--first", type=int, help="First element in the list of Pauli groups to be excecuted ")
    parser.add_argument("--last", type=int, help="last element in the list of Pauli groups to be excecuted ")
    args = parser.parse_args()

    # pauli_groups of commuting observables which can be measured simulatenously. 
    pauli_groups_file = args.Pauligroups
    first_element = args.first
    last_element = args.last

    # load the pauli groups file with 83 commuting groups for Guadalupe geometry
    pauli_groups = data_loader_json(pauli_groups_file)

    # The json file contains pauli strings in big endian notation
    # below convert those to little endian 
    expectations = convert_big_to_little_endian(pauli_groups)

    # loop over the elements of different commuting group. 
    for expectation in expectations[first_element:last_element]:
        meas_bases, over_head,circ_res_expt,perexpt = PER_step_with_pickle(perexpt =
                                                               per_experiment,
                                                               expectation = expectation,
                                                               samples = num_samples,
                                                               noise_strengths = noise_strengths_for_PER, backend=choosen_backend)
        
        print("Pauli group %d is being excecuted with the measurement basis %s "%(first_element,meas_bases))
        output_file = "expectation_all_%d.pickle"%first_element
        output_overhead = "overhead_all_%d.pickle"%first_element
        output_perexpt = "perexpt_all_%d.pickle"%first_element
        data_dump_pickle(output_data = circ_res_expt, output_file_name = output_file)
        data_dump_pickle(output_data = over_head, output_file_name = output_overhead)
        data_dump_pickle(output_data = perexpt, output_file_name = output_perexpt)

        first_element+=1
    return 

if __name__ == "__main__":
    main()


