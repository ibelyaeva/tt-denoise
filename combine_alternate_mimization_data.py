import numpy as np
import os
import configparser
from os import path
import pandas as pd
from collections import OrderedDict
import mri_draw_utils as mrd

results_path = "/work/project/cmsc655/results/alternate_min/"
richian_noise = "/work/project/cmsc655/results/alternate_min/richian"
guassian_noise = "/work/project/cmsc655/results/alternate_min/guassian/"
raleign_noise = "/work/project/cmsc655/results/alternate_min/raleign/"

godec_results = "/work/project/cmsc655/figures/godec/results/solution_go_dec.csv"

files = []
files.append("solution_cost_500.csv")
files.append("solution_cost_1000.csv")
files.append("solution_cost_1500.csv")
files.append("solution_cost_2000.csv")
files.append("solution_cost_2500.csv")
files.append("solution_cost_3500.csv")
files.append("solution_cost_4000.csv")

snr_level = []

snr_level.append(5)
snr_level.append(10)
snr_level.append(15)
snr_level.append(20)
snr_level.append(25)
snr_level.append(35)
snr_level.append(40)

def get_resuts_path(folder_path, result_path):
    result_path = os.path.join(folder_path, result_path)
    return result_path

def save_csv_by_path_adv(df, file_path, dataset_id, index = False):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path, index = index)
    
def read_dataset(folder_path, noise_type):
    
    datasets = []
    ctr = 0
    for item in files:
        dataset =  read_results(get_resuts_path(folder_path, item))
        dataset['noise_type'] = noise_type
        datasets.append(dataset)
        ctr = ctr + 1
        
    solution_datasets = pd.concat(datasets, axis = 0)
    return solution_datasets

def read_godec(folder_path, noise_type):
    
  
    dataset =  read_results(folder_path)
    dataset['noise_type'] = noise_type
    
    return dataset

def read_results(file_path):
    result = pd.read_csv(file_path, sep=',')
    return result


def aggregate_results():
    all_datasets = []
    solution_list = []
    richian_solution = read_dataset(richian_noise, 'richian')
    guassian_solution = read_dataset(guassian_noise, 'guassian')
    raleign_solution = read_dataset(raleign_noise, 'raleign')
      
    for s in  snr_level:
        subset=guassian_solution.loc[guassian_solution['snr'] == s]
        final_solution = subset.tail(1)
        solution_list.append(final_solution)
        
    for s in  snr_level:
        subset=richian_solution.loc[richian_solution['snr'] == s]
        final_solution = subset.tail(1)
        solution_list.append(final_solution)
        
    for s in  snr_level:
        subset=raleign_solution.loc[raleign_solution['snr'] == s]
        final_solution = subset.tail(1)
        solution_list.append(final_solution)
    
    solution_by_snr = pd.concat(solution_list, axis=0)
    solutuon_name = "alternate_minimization_solution_agg"
    
    save_csv_by_path_adv(solution_by_snr, results_path, solutuon_name, index = False)
    
    
def combine_datasets():
    col_names = ['k','snr','noise_type','low_rank_rse','sparse_rank_rse',' rel_solution_cost', 'solution_cost', 'initial_snr', 'corruption_error', 'solution_grad', 'solution_snr']
    all_datasets = []

    richian_solution = read_dataset(richian_noise, 'richian')
    all_datasets.append(richian_solution)
       
    guassian_solution = read_dataset(guassian_noise, 'guassian')
    all_datasets.append(guassian_solution)
    
    raleign_solution = read_dataset(raleign_noise, 'raleign')
    all_datasets.append(raleign_solution)
    
    final_solution = pd.concat(all_datasets, axis=0)
    solutuon_name = "alternate_minimization_solution"
    save_csv_by_path_adv(final_solution, results_path, solutuon_name, index = False)
    
def combine_datasets_with_godec():
    
    godec_res = read_godec(godec_results, 'richian')
    data = read_dataset(richian_noise, 'richian')
    
   
    subset=data.loc[data['snr'] == 25]
    final_solution = pd.concat([subset, godec_res], axis=1, join_axes=[subset.index])
  
    #all_datasets.append(richian_solution)
     
    solutuon_name = "alternate_minimization_solution_and_godec"
    
    save_csv_by_path_adv(final_solution, results_path, solutuon_name, index = False)
    
    
if __name__ == "__main__":
    #combine_datasets()
    #combine_datasets_with_godec()
    aggregate_results()
    
