import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
#X = pd.read_csv("data/cd_matrix_dA.csv").values
#print(X.shape)
X = np.random.rand(10,10)
#result = ges(X, score_func="local_score_CV_general")
result = ges(X, score_func="local_score_BDeu")

print(result)