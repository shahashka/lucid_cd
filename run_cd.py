import src.algs as algs
import numpy as np
import pandas as pd
import pickle
from cd_v_partition.fusion import screen_projections
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import time
celltype = "rpe1" # huvec
doses = ["A", "B", "C", "D", "E"]
cd_algs = [("dag_gnn", algs.dag_gnn_local_learn)]
            #("ges_non_param_tetrad", algs.ges_non_param_tetrad_local_learn),
            # ("pc_kci_tetrad", algs.pc_kci_tetrad_local_learn),
            # ("ges_bic_tetrad", algs.ges_bic_tetrad_local_learn),
            # ("pc_fisherz_tetrad", algs.pc_fisherz_tetrad_local_learn)]
domain_algs = [("genie3", algs.genie3_local_learn), ("GENELink", algs.GENELink_local_learn)]

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

def run_cd_partition():
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
            
            # Locally learn over partition
            # n_comms = len(custom_partition)
            # nthreads = n_comms  # each thread handles one partition
            results = []
            # chunksize = max(1, n_comms // nthreads)
            # with ProcessPoolExecutor(max_workers=nthreads) as executor:
            #     for result in executor.map(cd, subproblems, chunksize=chunksize):
            #         results.append(result)
            for s in subproblems:
                results.append(cd(s))
        
            # Create global graph 
            num_genes = len(df.columns) - 1
            superstructure = np.ones((num_genes+1, num_genes+1))
            global_net_non_param = screen_projections(superstructure, custom_partition, results, ss_subset=False, finite_lim=False, data=df.to_numpy())
            nx.write_gexf(global_net_non_param, f"./data/{celltype}/cd_{cd_name}_d{dose}.gexf")
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


def run_GENELink():
    for dose in doses:
        algs.GENELink_local_learn(dose)
        
if __name__ == "__main__":
    #run_BEELINE()
    run_cd_partition()
    # run_cd_no_partition()
    #run_GENELink()