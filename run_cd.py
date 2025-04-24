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
from cd_v_partition.causal_discovery import ges_local_learn
import torch
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

df = pd.read_csv("data/huvec/cd_matrix_dA.csv")
df_gene_expression = df.drop(columns=["radiation"])
num_genes = len(df_gene_expression.columns)
df_gene_expression = StandardScaler().fit_transform(df_gene_expression)
df_gene_expression = pd.DataFrame(data=df_gene_expression)
df_learn = pd.DataFrame(data=df_gene_expression)
df_learn['radiation'] = df['radiation']

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
    data = data.drop(columns=['target'])
    result = pc(data.values, alpha=0.05, indep_test='fastkci')
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
    d = data.shape[1]
    X = data.values
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    adj = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    return adj


#global_adj = ges_local_learn([superstructure, df_learn], use_skel=True)
start = time.time()
global_adj_dagma = dagma_nonlinear_local_learn([None, df_learn], use_skel=False)
print(f"DAGMA took {time.time() - start}(s) and learned {np.sum(global_adj_dagma!=0)} edges")

start = time.time()
global_adj_gnn = dag_gnn_local_learn([None, df_learn], use_skel=False)
print(f"DAG-GNN took {time.time() - start}(s) and learned {np.sum(global_adj_gnn!=0)} edges")

start = time.time()
global_adj_notears = notears_mlp_local_learn([None, df_learn], use_skel=False)
print(f"NOTEARS took {time.time() - start}(s) and learned {np.sum(global_adj_notears!=0)} edges")


# Correlation matrix with Permutation testing
# df_gene_expression = df.drop(columns=["radiation"])
# num_genes = len(df_gene_expression.columns)
# df_gene_expression = StandardScaler().fit_transform(df_gene_expression)
# df_gene_expression = pd.DataFrame(data=df_gene_expression)
# corr_mat = df_gene_expression.corr('pearson', numeric_only=True).to_numpy()
# print(np.min(np.abs(corr_mat)))
# np.fill_diagonal(corr_mat, 0)
# print(np.max(corr_mat), np.argmax(corr_mat))
# random_corr_coef = []
# for i in range(10):
#     shuffled_array = df_gene_expression.values
#     [np.random.shuffle(x) for x in shuffled_array]
#     shuffled_final_data_set = pd.DataFrame(data=shuffled_array)
#     shuffle_corr_mat = shuffled_final_data_set.corr('pearson', numeric_only=True).to_numpy()
#     np.fill_diagonal(shuffle_corr_mat, 0)
#     print(np.max(shuffle_corr_mat), np.argmax(shuffle_corr_mat))
#     random_corr_coef.append(np.max(shuffle_corr_mat))
# print(random_corr_coef)
# ci_interval = st.t.interval(0.95, len(random_corr_coef)-1, 
#                             loc=np.mean(random_corr_coef), 
#                             scale=st.sem(random_corr_coef))
# print(ci_interval)
# cutoff = ci_interval[1]

# corr_mat[corr_mat<=cutoff] = 0
# corr_mat[corr_mat>cutoff] = 1
# print(f"Superstructure contains {np.sum(corr_mat)} edges which is \
#         {np.sum(corr_mat)/(corr_mat.shape[1]**2)} fraction of all possible edges")

# skeleton = [(i,j) for (i,j) in itertools.product(range(num_genes), range(num_genes)) if corr_mat[i,j] ==1]
# skeleton += [(num_genes,i) for i in range(num_genes)]
# print(len(skeleton))
# # This implementation of GES is painfully slow, the graph operations (converting a DAG to a PDAG take forever for large graphs)
# result = ges(StandardScaler().fit_transform(X), maxP=10, score_func="local_score_CV_general", skeleton=skeleton)
