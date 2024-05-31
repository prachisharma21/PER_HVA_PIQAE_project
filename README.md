In this project, we use error mitigation technique PER (Probabilistic Error Reduction) in order to find the ground state energies using VQE (Variational Quantum Eigensolver) integrated with Quantum Krylov subpsace expansion method. The paper titled ["Quantum subspace expansion in the presence of hardware noise"](https://arxiv.org/abs/2404.09132) is on arxiv. The model studied is Mixed field Ising model (MFIM) with the heavyhex geometry of IBM backends Quito (5-qubit device) and Guadalupe (16-qubit device).  

Details of the code in this repository. It basically does two things. 
1. VQE.py performs the VQE step based on HVA(Hamiltonian Variational Ansatz) for  a Mixed Field Ising Model and outputs the optimized parameters 
2. PER.py performs the error mitigation using PER method for the all the Pauli strings for Krylov subspace expansion on the fake backend of Quito and Guadalupe. All commuting Pauli strings are grouped together to reduce the number of measurements. 
3. post_processing.py performs the analysis to derive the ZNE expectation values based on the PER mitigated values from different noise strengths. 

4. helper.py has some simple helper functions and classes for ease. 
5. model_hamiltonian.py store some useful functions for the MFIM model. 
5. circuit_builder.py includes the class to build the VQE parameterized circuits for any number of layers of HVA ansatz. 


