import src.algs as algs
import numpy as np
import pandas as pd
import pickle
from cd_v_partition.fusion import screen_projections, no_partition_postprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial 
import networkx as nx
import time
import os
from tqdm import tqdm
from src.GENELink.Code.Train_Test_Split import Hard_Negative_Specific_train_test_val
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
celltype = "rpe1_experiment2" #"huvec" 
doses = ["F", "G", "H", "I", "J"]#["A", "B", "C", "D", "E"] #
cd_algs = [("dag_gnn", algs.dag_gnn_local_learn)]#, ("lingam_tetrad", algs.direct_lingam_tetrad_local_learn)]
            #("ges_bic_tetrad", algs.ges_bic_tetrad_local_learn) ]
            #("ges_non_param_tetrad", algs.ges_non_param_tetrad_local_learn),
            # ("pc_kci_tetrad", algs.pc_kci_tetrad_local_learn),
            # , ("pc_fisherz_tetrad", algs.pc_fisherz_tetrad_local_learn)
              

domain_algs = [("genie3_tf", algs.genie3_local_learn)]#, ("GENELink", algs.GENELink_local_learn)]

def partition_problem(partition: dict, data: pd.DataFrame):
    sub_problems = []
    k = list(partition.keys())
    k.sort()
    for i in k:
        sub_nodes = partition[i]
        sub_nodes = list(sub_nodes)
        sub_data = data[sub_nodes] 
        sub_problems.append(sub_data)
    return sub_problems

def run_cd_partition(filter_by_variance = False, bootstrap=False):
    for dose in doses:
        for cd_name, cd in cd_algs:
            print(f'Running {cd_name} for dose {dose}')
            start = time.time()
            # Load gene epxression 
            df = pd.read_csv(f"data/{celltype}/cd_tpm_matrix_d{dose}.csv")
            
            # Load curated partition, add radiation to each subset
            with open(f"./data/{celltype}/cd_partition_d{dose}_new.pickle", 'rb') as f:
                custom_partition = pickle.load(f)
            for i, comm in custom_partition.items():
                comm += ['radiation']
            subproblems = partition_problem(custom_partition, df)
            print(len(subproblems), custom_partition.keys())
            # Locally learn over partition
            # n_comms = len(custom_partition)
            # nthreads = n_comms  # each thread handles one partition
            results = []
            gene_names = []
            # chunksize = max(1, n_comms // nthreads)
            # with ProcessPoolExecutor(max_workers=nthreads) as executor:
            #     for result in executor.map(cd, subproblems, chunksize=chunksize):
            #         results.append(result)
            for i,s in enumerate(subproblems):
                if filter_by_variance:
                    variance = np.var(np.log1p(s.to_numpy()),axis=0)
                    filtered_inds = np.where(variance>0.05)
                    filtered_genes = list(s.columns[filtered_inds])
                    gene_names.append(filtered_genes)
                    s = s[filtered_genes]
                if cd_name == 'genie3' or cd_name=='genie3_tf':
                    threshold=0
                else:
                    threshold=0.3
                if bootstrap and (dose=='F' and i <=2):
                    results.append(bootstrap_cd(i, cd_name, cd, s, dose,threshold=threshold, n_bootstraps=10))
                # else:
                #     results.append(bootstrap_cd(i, cd_name, cd, s, dose,threshold=threshold, n_bootstraps=1))

                # filtered_genes = pd.read_csv(f"./data/{celltype}/bootstrap_graphs5/part_{i}_gene_names_{cd_name}_d{dose}.csv")
                # gene_names.append(filtered_genes)
                # arr = np.load(f"./data/{celltype}/bootstrap_graphs5/part_{i}_edge_prob_{cd_name}_d{dose}.npy")
                # threshold=0.5
                # arr[arr < threshold] = 0
                # results.append(arr)
            # Create global graph 
            # for i, (r,g) in enumerate(zip(results, gene_names)):
            #     np.save(f"./data/{celltype}/bootstrap_graphs5/part_{i}_edge_prob_{cd_name}_d{dose}.npy", r)
            #     gene_ids = pd.DataFrame(g)
            #     gene_ids.to_csv(f"./data/{celltype}/bootstrap_graphs5/part_{i}_gene_names_{cd_name}_d{dose}.csv")

            num_genes = len(df.columns) - 1
            superstructure = np.ones((num_genes+1, num_genes+1))
            #global_net_non_param = screen_projections(superstructure, custom_partition, results, ss_subset=False, finite_lim=False, data=df.to_numpy())
            #global_net_non_param = nx.from_numpy_array(A_mat, create_using=nx.DiGraph)
            #node_ids = s.columns
            # global_net_non_param = nx.relabel_nodes(
            #     global_net_non_param,
            #     mapping=dict(zip(np.arange(len(node_ids)), node_ids)),
            #     copy=True,
            # )
            file_path = f"./data/{celltype}/bootstrap_graphs2/cd_{cd_name}_d{dose}.gexf" #if bootstrap else f"./data/{celltype}/cd_{cd_name}_d{dose}.gexf"
            #nx.write_gexf(global_net_non_param, file_path)
            print(f'Done running {cd_name} for dose {dose} in {time.time() - start}(s)')
            
def run_cd_no_partition():
    for dose in doses:
        for cd_name, cd in cd_algs:
            print(f'Running {cd_name} for dose {dose}')
            start = time.time()
            # Load gene epxression 
            df = pd.read_csv(f"data/{celltype}/cd_matrix_d{dose}.csv")        
            ground_truth = nx.read_gexf(f"./data/{celltype}/cd_subgraph_d{dose}.gexf" )
            
            global_adj = cd(df[list(ground_truth.nodes)+['radiation']])
            global_net = nx.from_numpy_array(global_adj, create_using=nx.DiGraph)
            global_net = nx.relabel_nodes(global_net, dict(zip(np.arange(len(global_net.nodes)), list(ground_truth.nodes)+['radiation'])))

            nx.write_gexf(global_net, f"./data/{celltype}/cd_{cd_name}_d{dose}_np.gexf")
            print(f'Done running {cd_name} for dose {dose} in {time.time() - start}(s)')

def run_BEELINE():
    df_bl_GENELink_1000 = pd.read_csv('./src/GENELink/Code/Demo/hESC/TFs+1000/BL--ExpressionData.csv')
    df_bl_GENELink_1000_net = pd.read_csv('./src/GENELink/Code/Demo/hESC/TFs+1000/BL--network.csv')
    print(df_bl_GENELink_1000)
    print(df_bl_GENELink_1000_net)
    num_samples = df_bl_GENELink_1000.shape[1] - 1
    num_genes = df_bl_GENELink_1000.shape[0]   
    data = df_bl_GENELink_1000.iloc[:,1:num_samples+1].T.values
    print(data, data.shape)
    df_bl_GENELink_1000_learn = pd.DataFrame(data=data)
    df_bl_GENELink_1000_learn.columns = df_bl_GENELink_1000.iloc[:,0]
    global_adj_bl = algs.dag_gnn_local_learn(df_bl_GENELink_1000_learn)
    ground_truth_net = nx.from_edgelist(df_bl_GENELink_1000_net.values.tolist(), create_using=nx.DiGraph)
    print(ground_truth_net)
    sorted_nodes = sorted(ground_truth_net.out_degree(), key=lambda x: x[1], reverse=True)
    # Print nodes and their degree
    print("Ground truth graph nodes sorted by degree:")
    for node, degree in sorted_nodes:
        print(f"{node}: degree {degree}")
        
    global_net_bl = nx.from_numpy_array(global_adj_bl, create_using=nx.DiGraph)
    global_net_bl = nx.relabel_nodes(global_net_bl, dict(zip(np.arange(len(global_net_bl.nodes)), list(df_bl_GENELink_1000_learn.columns))))
    print(global_net_bl)
    true_positive = 0
    for edge in ground_truth_net.edges:
        if global_net_bl.has_edge(edge[0], edge[1]):
            true_positive += 1
            print(edge)
    print(f"Number of true positive edges {true_positive}")
    sorted_nodes = sorted(global_net_bl.out_degree(), key=lambda x: x[1], reverse=True)
    # Print nodes and their degree
    print("Estimated graph nodes sorted by degree:")
    for node, degree in sorted_nodes:
        print(f"{node}: degree {degree}")
    nx.write_gexf(global_net_bl, f"./data/BEELINE/cd_dag_gnn_1000.gexf")


def run_GENELink(bootstrap=False, with_radiation=False):
    path = f'./data/{celltype}/GENELink_data_files'
    n_bootstrap = 10
    for dose in doses:
        start = time.time()
        exp_file = f"{path}/GeneExpression_key_genes_{dose}.csv"            
        data = pd.read_csv(exp_file,index_col=0)
        if with_radiation:
            rad_column = pd.read_csv(f"{path}/GeneExpression_d{dose}.csv",index_col=0)['radiation']
            data['radiation'] = rad_column
        data = data.T
        if bootstrap:
            for i in range(n_bootstrap):
                indices = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
                data_boot = data.iloc[indices]
                algs.GENELink_local_learn(data_boot, dose, seed=i)
        else:
            algs.GENELink_local_learn(data, dose, seed="")
        print(f"Training GENELink took {time.time() - start} (s)")

# Bootstrapping function
def boot_func(i, X, cd, celltype, cd_name, threshold, dose, part):
    print(i)
    indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
    X_boot = X.iloc[indices]
    if cd == algs.genie3_local_learn:
        A_boot = cd(X_boot, celltype, dose, i)
    else:
        A_boot = cd(X_boot, i)
    edge_counts = (np.abs(A_boot) > threshold).astype(int)
    np.save(f"./data/{celltype}/bootstrap_graphs3/{cd_name}_dose_{dose}_part_{part}_boot_{i}.npy", A_boot)
    return A_boot,edge_counts

def bootstrap_cd(part,cd_name, cd, X, dose, n_bootstraps=20, threshold=0.3):
    n_genes = X.shape[1]
    # edge_counts = np.zeros((n_genes, n_genes)) 
 
    partial_function = partial(boot_func, X=X, cd=cd,celltype=celltype, cd_name=cd_name, threshold=threshold, dose=dose, part=part)   
    # with multiprocessing.Pool() as pool, tqdm(total=n_bootstraps, desc="Bootstrapping") as pbar:
    #     for result in pool.imap(partial_function, range(0, n_bootstraps)):
    #         pbar.update()
    #         pbar.refresh()
            # edge_counts += result[1]
    for i in tqdm(range(n_bootstraps), desc="Bootstrapping..."):
        A_boot, ec = partial_function(i)  
    #     edge_counts += ec
    # edge_probs = edge_counts / n_bootstraps
    # return edge_probs # For now measure edge stability 

def run_combined_cd():
    cd_name="dag_gnn"
    cd = algs.dag_gnn_local_learn
    threshold=0.3
    df = pd.read_csv(f"data/{celltype}/cd_tpm_matrix_combined_dose_rate.csv")
    with open(f"./data/{celltype}/cd_partition_combined.pickle", 'rb') as f:
        custom_partition = pickle.load(f)
    for i, comm in custom_partition.items():
        comm += ['week']
        comm += ['dose_rate']
    subproblems = partition_problem(custom_partition, df)
    print(len(subproblems), custom_partition.keys())
    for i,s in enumerate(subproblems):
        bootstrap_cd(i, cd_name, cd, s,threshold=threshold, n_bootstraps=10, dose="combined")

def train_test_split_genelink():
    for d in doses:
        label_file = f"data/{celltype}/GENELink_data_files/Label_{d}.csv"
        target_file = f"data/{celltype}/GENELink_data_files/Target_{d}.csv"
        tf_file = f"data/{celltype}/GENELink_data_files/TF_{d}.csv"
        train_set_file = f"data/{celltype}/GENELink_data_files/Train_set_{d}.csv"
        val_set_file = f"data/{celltype}/GENELink_data_files/Validation_set_{d}.csv"
        test_set_file = f"data/{celltype}/GENELink_data_files/Test_set_{d}.csv"
        _  = Hard_Negative_Specific_train_test_val(label_file=label_file, 
                                                                Gene_file=target_file, 
                                                                TF_file=tf_file,
                                                                train_set_file= train_set_file,
                                                                val_set_file= val_set_file,
                                                                test_set_file = test_set_file,
                                                                ratio=0.67,
                                                                p_val=0.5)
if __name__ == "__main__":
    #run_BEELINE()
    #run_cd_partition(filter_by_variance=False, bootstrap=True)
    # run_cd_no_partition()
    #run_GENELink(bootstrap=False, with_radiation=False)
    run_combined_cd()
    #train_test_split_genelink()