import src.algs as algs
import numpy as np
import pandas as pd
import pickle
from cd_v_partition.fusion import screen_projections
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import time

doses = ["A", "B", "C", "D", "E"]
cd_algs = [("dag_gnn", algs.dag_gnn_local_learn),
            ("ges_non_param_tetrad", algs.ges_non_param_tetrad_local_learn),
            ("pc_kci_tetrad", algs.pc_kci_tetrad_local_learn),
            ("ges_bic_tetrad", algs.ges_bic_tetrad_local_learn),
            ("pc_fisherz_tetrad", algs.pc_fisherz_tetrad_local_learn)]

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
            df = pd.read_csv(f"data/huvec/cd_matrix_d{dose}.csv")
            
            # Load curated partition, add radiation to each subset
            with open(f"./data/huvec/cd_partition_d{dose}_new.pickle", 'rb') as f:
                custom_partition = pickle.load(f)
            for i, comm in custom_partition.items():
                comm += ['radiation']
            subproblems = partition_problem(custom_partition, df)
            
            # Locally learn over partition
            n_comms = len(custom_partition)
            nthreads = n_comms  # each thread handles one partition
            results = []
            chunksize = max(1, n_comms // nthreads)
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                for result in executor.map(cd, subproblems, chunksize=chunksize):
                    results.append(result)

            # Create global graph 
            num_genes = len(df.columns) - 1
            superstructure = np.ones((num_genes+1, num_genes+1))
            global_net_non_param = screen_projections(superstructure, custom_partition, results, ss_subset=False, finite_lim=False, data=df.to_numpy())

            nx.write_gexf(global_net_non_param, f"./data/huvec/cd_{cd_name}_d{dose}.gexf")
            print(f'Done running {cd_name} for dose {dose} in {time.time() - start}(s)')
            
def run_cd_no_partition():
    for dose in doses:
        for cd_name, cd in cd_algs:
            print(f'Running {cd_name} for dose {dose}')
            start = time.time()
            # Load gene epxression 
            df = pd.read_csv(f"data/huvec/cd_matrix_d{dose}.csv")        
            ground_truth = nx.read_gexf(f"./data/huvec/cd_subgraph_d{dose}.gexf" )
            
            global_adj = cd([None, df[list(ground_truth.nodes)+['radiation']]], use_skel=False)
            global_net = nx.from_numpy_array(global_adj, create_using=nx.DiGraph)
            global_net = nx.relabel_nodes(global_net, dict(zip(np.arange(len(global_net.nodes)), list(ground_truth.nodes)+['radiation'])))

            nx.write_gexf(global_net, f"./data/huvec/cd_{cd_name}_d{dose}_np.gexf")
            print(f'Done running {cd_name} for dose {dose} in {time.time() - start}(s)')

            
if __name__ == "__main__":
    run_cd_partition()
    run_cd_no_partition()