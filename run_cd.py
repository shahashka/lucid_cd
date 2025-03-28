import numpy as np
import pandas as pd
import scipy.stats as st
from causallearn.search.ScoreBased.GES import ges
import itertools
from sklearn.preprocessing import StandardScaler
# X = np.random.normal(0,1,(1000, 1))
# Y = X + np.random.normal(0,0.1,(1000, 1))
# data = np.column_stack([X,Y])
# print(data.shape)
# result = ges(data, maxP=10, score_func="local_score_CV_general", skeleton=None)
# print(result['G'])
df = pd.read_csv("data/cd_matrix_dA.csv")
X = df.values
print(X.shape)
skeleton = itertools.product(range(X.shape[1]), range(X.shape[1]-1)) 

# Correlation matrix with Permutation testing
df_gene_expression = df.drop(columns=["radiation"])
num_genes = len(df_gene_expression.columns)
df_gene_expression = StandardScaler().fit_transform(df_gene_expression)
df_gene_expression = pd.DataFrame(data=df_gene_expression)
corr_mat = df_gene_expression.corr('pearson', numeric_only=True).to_numpy()
print(np.min(np.abs(corr_mat)))
np.fill_diagonal(corr_mat, 0)
print(np.max(corr_mat), np.argmax(corr_mat))
random_corr_coef = []
for i in range(10):
    shuffled_array = df_gene_expression.values
    [np.random.shuffle(x) for x in shuffled_array]
    shuffled_final_data_set = pd.DataFrame(data=shuffled_array)
    shuffle_corr_mat = shuffled_final_data_set.corr('pearson', numeric_only=True).to_numpy()
    np.fill_diagonal(shuffle_corr_mat, 0)
    print(np.max(shuffle_corr_mat), np.argmax(shuffle_corr_mat))
    random_corr_coef.append(np.max(shuffle_corr_mat))
print(random_corr_coef)
ci_interval = st.t.interval(0.95, len(random_corr_coef)-1, 
                            loc=np.mean(random_corr_coef), 
                            scale=st.sem(random_corr_coef))
print(ci_interval)
cutoff = ci_interval[1]

corr_mat[corr_mat<=cutoff] = 0
corr_mat[corr_mat>cutoff] = 1
print(f"Superstructure contains {np.sum(corr_mat)} edges which is \
        {np.sum(corr_mat)/(corr_mat.shape[1]**2)} fraction of all possible edges")

skeleton = [(i,j) for (i,j) in itertools.product(range(num_genes), range(num_genes)) if corr_mat[i,j] ==1]
skeleton += [(num_genes,i) for i in range(num_genes)]
print(len(skeleton))
# This implementation of GES is painfully slow, the graph operations (converting a DAG to a PDAG take forever for large graphs)
result = ges(StandardScaler().fit_transform(X), maxP=10, score_func="local_score_CV_general", skeleton=skeleton)
