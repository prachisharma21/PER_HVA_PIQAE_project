import pickle
import json
sys.path.append("/home/users/psharma/Documents/AutomatedPERTools-main/pauli_lindblad_per")
from tomography.experiment import SparsePauliTomographyExperiment as tomography
from primitives.pauli import QiskitPauli


def data_loader_expectation(input_file = "output_0.pickle"):
    with open(input_file,"rb") as f:
        circ_res = pickle.load(f)
    return circ_res

def data_loader_pauligroups(input_file = 'pauli_groups.json'):
    file = open(input_file)
    data = json.load(file)
    return data 

pauli_groups = data_loader_pauligroups()
pauli_groups = pauli_groups[11:13]
dic_of_dic = {}
for i,pauli_group in enumerate(pauli_groups):
    big_to_little_endian_paulis = ["".join(reversed([p for p in pauli]))for pauli in pauli_group]
    expectation_data = data_loader_expectation(input_file='/home/prachi/expectation_%d.pickle'%i)
    dic_res = {}
    for pauli in big_to_little_endian_paulis:
        #print(pauli)
        #print(type(pauli))
        big_endian_pauli = "".join(reversed([p for p in pauli]))
        #print(big_endian_pauli)
        res = expectation_data[0].get_result(pauli).expectation
              
        dic_res[big_endian_pauli] = res 
    dic_of_dic["%d"%i] = dic_res

print(dic_of_dic)

with open("res_dict_100_PER_Guadalupe.pickle", "wb") as f:
    pickle.dump(dic_of_dic, f)
