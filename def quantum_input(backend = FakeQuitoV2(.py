def quantum_input(backend = FakeQuitoV2()):
    """ function is to get the basic essentials inputs for the quantum device from the user """    
    backend = backend 
    num_qubits = backend.num_qubits
    return backend, num_qubits

def vqeLayer_FakeQuito(theta_ZZ, theta_Z, theta_X, num_qubits = FakeQuitoV2.num_qubits):
    """ VQE layer for the FakeQuito geometry using all qubits and native connectivity"""
    vqeLayer = QuantumCircuit(num_qubits)
    # Choosen bond pairs according to the native qubit connectivity of the backend
    bonds_1 = [[0, 1], [3, 4]]  # these bond pairs should be accessible as well-- they will be needed for hamiltonian expectation for eg. 
    bonds_2 = [[1, 2]] # could be added to the main QuantumSystem class as they are dependent on the backend geometry
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

def PER_step(expt = experiment, expectation = expectations, samples = 10, noise_strengths = [0, 0.25, 0.5]):
    perexp = experiment.create_per_experiment(circuits_vqe_no_meas)
    perexp.generate(expectations = expectations, samples = samples, noise_strengths = noise_strengths)
    print(perexp.meas_bases)
    perexp.run(executor)
    circuit_results = perexp.analyze() #analyze the circuits to carry the above process out for each circuit, depth, and expectation value,
    #and apply vZNE
    return circuit_results

# Having intial_layout is better---as then you can choose not to use all qubits. 
def makevqeCircuit(theta_ZZ, theta_Z, theta_X, initial_layout = initial_layout,measure = False, meas_basis = "Z"):
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
        vqeCircuit = vqeCircuit.compose(vqeL)
        vqeCircuit.barrier()
    #vqeCircuit.measure_all()
    
    if measure == True:
        if meas_basis == "Z":
            vqeCircuit.measure_all()
            transpiled = transpile(vqeCircuit, backend, initial_layout = initial_layout)
        elif meas_basis =="X":
            vqeCircuit.h(range(num_qubits))
            vqeCircuit.measure_all()
            transpiled = transpile(vqeCircuit, backend, initial_layout = initial_layout)
        else: 
             print("Measurement not defined")    # Y-measurements can be added
    else:
        transpiled = transpile(vqeCircuit, backend, initial_layout = initial_layout)
    return transpiled

def QASM_simulation(shots = 1000, sim_backend = "FakeQuito", initial_layout = initial_layout):# quantum_circuit =circuits_vqe_no_meas[0].compose  )
    # apply measurements 
    circuit_with_meas = makevqeCircuit(theta_ZZ_L_1, theta_Z_L_1, theta_X_L_1, initial_layout = initial_layout,measure = True, meas_basis = "Z")
    count = Aer.get_backend('qasm_simulator').run(circuit_with_meas, shots=shots).result().get_counts()
    return count





#
# *****************************************************************************************************************************************************

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


class CircuitBuilder(Quantum_system):
    def __init__(self,backend , initial_layout , geometry):
        super().__init__(backend,initial_layout,geometry)
        #print("backend here is ", self.backend)
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

