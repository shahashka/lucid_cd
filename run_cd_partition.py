import numpy as np
import pandas as pd
import scipy.stats as st
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
import itertools
from sklearn.preprocessing import StandardScaler
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from dagma.linear import DagmaLinear
from castle.algorithms import NotearsNonlinear, DAG_GNN
from notears.notears.nonlinear import NotearsMLP, notears_nonlinear
import torch
import os
import pickle
from cd_v_partition.fusion import screen_projections, fusion
import functools
from concurrent.futures import ProcessPoolExecutor
from cd_v_partition.causal_discovery import ges_local_learn
import networkx as nx
from netgraph import Graph
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def dagma_nonlinear_local_learn(subproblem, use_skel):
    skel, data = subproblem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = data.shape[1]
    X = data.values
    X_tensor = torch.from_numpy(X).type(torch.double).to(device)
    eq_model = DagmaMLP(dims=[d, np.min([10, d-1]), 1], bias=True, dtype=torch.double) # create the model for the structural equations, in this case MLPs
    eq_model = eq_model.to(device)
    model = DagmaNonlinear(eq_model, dtype=torch.double) # create the model for DAG learning
    # adj = model.fit(X_tensor, lambda1=0.02, lambda2=0.005) # fit the model with L1 reg. (coeff. 0.02) and L2 reg. (coeff. 0.005)
    adj = model.fit(X_tensor, lambda1=0.001, lambda2=0.0001) # fit the model with L1 reg. (coeff. 0.02) and L2 reg. (coeff. 0.005)

    return adj

def ges_non_param_local_learn(subproblem, use_skel):
    skel, data = subproblem
    data = data.drop(columns=['target'])
    print(data.columns)
    result = ges(data.values, maxP=10, node_names=data.columns, score_func="local_score_CV_general")
    return result
def pc_kci_local_learn(subproblem, use_skel):
    skel, data = subproblem
    result = pc(data.values, alpha=0.05, indep_test='kci')
    return result
def dag_gnn_local_learn(subproblem, use_skel):
    skel, data = subproblem
    d = data.shape[1]
    X = data.values
    model = DAG_GNN(device_type="gpu")
    model.learn(X)
    return model.causal_matrix
    
def notears_mlp_local_learn(subproblem, use_skel):
    skel, data = subproblem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = data.shape[1]
    X = data.values
    X_tensor = torch.from_numpy(X).type(torch.double).to(device)
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    model.to(device)
    #print(model.fc1_neg.weight.device, X.device)
    adj = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    return adj
    
def partition_problem(partition: dict, structure: np.ndarray, data: pd.DataFrame):
    sub_problems = []
    k = list(partition.keys())
    k.sort()
    for i in k:
        sub_nodes = partition[i]
        sub_structure = np.ones((len(sub_nodes), len(sub_nodes)))
        sub_nodes = list(sub_nodes)
        sub_data = data[sub_nodes] 
        sub_problems.append((sub_structure, sub_data))
    return sub_problems

def main():    
    doses = ["A", "B", "C", "D", "E"]
    cd_algs = [("dag_gnn", dag_gnn_local_learn)]#, ("ges_non_param", ges_non_param_local_learn), ("pc_kci", pc_kci_local_learn)]
    for dose in doses:
        for cd_name, cd in cd_algs:
            print(f'Running {cd_name} for dose {dose}')
            # Load gene epxression 
            df = pd.read_csv(f"data/huvec/cd_matrix_d{dose}.csv")
            df_gene_expression = df.drop(columns=["radiation"])
            num_genes = len(df_gene_expression.columns)
            
            # Load curated partition, add radiation to each subset
            with open(f"./data/huvec/cd_partition_d{dose}_new.pickle", 'rb') as f:
                custom_partition = pickle.load(f)
            for i, comm in custom_partition.items():
                comm += ['radiation']
            superstructure = np.ones((num_genes+1, num_genes+1))
            subproblems = partition_problem(custom_partition, superstructure, df)
            
            # Locally learn over partition
            #Setup thread level parallelism
            func_partial = functools.partial(
                cd,
                use_skel=False,
            )
            n_comms = len(custom_partition)
            nthreads = n_comms  # each thread handles one partition
            results = []
            chunksize = max(1, n_comms // nthreads)
            with ProcessPoolExecutor(max_workers=nthreads) as executor:
                for result in executor.map(func_partial, subproblems, chunksize=chunksize):
                    results.append(result)

            # Create global graph 
            global_net_non_param = screen_projections(superstructure, custom_partition, results, ss_subset=False, finite_lim=False, data=df.to_numpy())
            print(len(global_net_non_param.edges()))
            print(len(global_net_non_param.nodes()))

            # Compare with 'ground truth' from TRRUST, htFTarget, STRING
            ground_truth = nx.read_gexf(f"./data/huvec/cd_subgraph_d{dose}.gexf" )
            print(ground_truth)
            true_positive = 0
            for edge in global_net_non_param.edges:
                if  ground_truth.has_edge(edge[0], edge[1]):
                    true_positive += 1
                    print(edge)
            print(f"True positives from background knowledge: {true_positive}")

            # Compare curated radiation biology specific edges from papers (some of these are from IPA)
            radbio_df = pd.read_csv("./data/prior_knowledge/radiation_edges.txt")
            radbio_net = nx.DiGraph({edge for edge in zip(radbio_df['node1'], radbio_df['node2'])})
            true_positive=0
            for edge in radbio_net.edges:
                if  global_net_non_param.has_edge(edge[0], edge[1]):
                    true_positive += 1
                    print(edge)
            print(f"True positives from curated radiation knowledge: {true_positive}")
            
            nx.write_gexf(global_net_non_param, f"./data/huvec/cd_{cd_name}_d{dose}.gexf")