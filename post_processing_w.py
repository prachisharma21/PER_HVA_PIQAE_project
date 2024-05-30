import numpy as np
from scipy.optimize import curve_fit
from helper import *

def fit(noise =[], expect = None,sigmas = None):
    """Perform an exponential fit of the expectation value as a function of the noise strength and provided standard deviations.    
    - In the large noise limit,
    the noise is completely depolarizing, and all expectation values tend toward zero.
    """ 
    expfit = lambda x,a,b: a*np.exp(b*x)
    popt, pcov = curve_fit(expfit, noise, expect, sigma = sigmas, bounds = [(-1.5, -1),(1.5,0)]) #b > 0 is enforced. Technically |a| <= 1 could also be enforced
    a,b = popt
    p_cov = pcov
    ZNE_expectation = a
    return a,b,p_cov



##################################################################################
pauli_groups_file = "pauli_groups.json"
pauli_groups = data_loader_json(pauli_groups_file)
#pauli_groups = pauli_groups[82:83]

dic_of_dic_1_w_std = {}
dic_of_dic_w_stdev = {}



for i,pauli_group in enumerate(pauli_groups):

    big_to_little_endian_paulis = ["".join(reversed([p for p in pauli]))for pauli in pauli_group]

    print("Performing pauli_group %d"%i)
    #expectation_data_all  = data_loader_expectation(input_file='/home/prachi/expectation_data_all/expectation_all/expectation_all_82.pickle')
    expectation_data_all  = data_loader_pickle(input_file='/home/prachi/expectation_data_all/expectation_all/expectation_all_%d.pickle'%i)
    dic_res_noise_1 = {}
    dic_res_SV ={}
    dic_res_fit = {}
    dic_for_kse = {}

    for pauli in big_to_little_endian_paulis:
        big_endian_pauli = "".join(reversed([p for p in pauli]))
        res_noise_strengths_all = expectation_data_all[0].get_result(pauli).get_strengths()#[1:] # excluding zero noise 
        res_all_expectation_all = expectation_data_all[0].get_result(pauli).get_expectations()#[1:]  # excluding zero noise result
        res_all_expectation_list = expectation_data_all[0].get_result(pauli).get_list_of_expectation() # tdictionary with noise as key and values are list fo expectations for the noise
        res_all_mean = []
        res_all_std = []

        for ns in res_noise_strengths_all:
            res_all_mean.append(np.mean(res_all_expectation_list[ns]))
            res_all_std.append(np.std(res_all_expectation_list[ns]))
            if ns == 1:
                #print(ns)
                res_1_mean = np.mean(res_all_expectation_list[ns])
                res_1_std = np.std(res_all_expectation_list[ns])

      
        a,b,cov = fit(noise = res_noise_strengths_all, expect = res_all_mean,sigmas =res_all_std )
        perr = np.sqrt(np.diag(cov)) # calculating standard deviation from the variance 

        dic_res_fit[big_endian_pauli] = (a,perr[0]) # zero because the std deviation is only the expectation values given by first parameter of the fit 
        dic_res_noise_1[big_endian_pauli] = (res_1_mean,res_1_std)



    dic_of_dic_w_stdev["%d"%i] = dic_res_fit
    dic_of_dic_1_w_std["%d"%i] = dic_res_noise_1

data_dump_pickle(output_data=dic_of_dic_w_stdev,output_file_name="res_dict_1000_noise_all_PER_Guadalupe_w_correct_std.pickle")
data_dump_pickle(output_data=dic_of_dic_1_w_std,output_file_name="res_dict_1000_noise_1_PER_Guadalupe_w_correct_std.pickle")


