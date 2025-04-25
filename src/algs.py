import numpy as np
import os
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from castle.algorithms import NotearsNonlinear, DAG_GNN
from src.GENIE3 import GENIE3

os.environ["JAVA_HOME"] = "/homes/shahashka/lucid_cd/amazon-corretto-21.0.7.6.1-linux-x64/lib/"
import pytetrad.tools.TetradSearch as py_ts
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import jpype
from jpype import JImplements, JOverride
import importlib.resources as importlib_resources
jar_path = importlib_resources.files('pytetrad').joinpath('resources','tetrad-current.jar')
jar_path = str(jar_path)
if not jpype.isJVMStarted():
    try:
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[jar_path])
    except OSError:
        print("can't load jvm")
        pass

import pandas as pd

import pytetrad.tools.translate as tr

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.search.score as sc

try:
    from causallearn.score.LocalScoreFunction import local_score_marginal_general
    from causallearn.score.LocalScoreFunction import local_score_cv_general
except ImportError as e:
    print('Could not import a causal-learn module: ', e)

# Can use this as a template for defining scores in Python for use with
# Java Tetrad algorithms.
@JImplements(sc.Score)
class Bgs:
    def __init__(self, df):
        self.df = df
        self.data = df.values
        self.parameters = {"kfold": 10, "lambda": 0.01}

        # pick a score: bug in marginal_general?
        # self.score = local_score_marginal_general
        self.score = local_score_cv_general

        # these scores are expensive, so caching seems pertinent...
        self.cache = {}

        self.variables = util.ArrayList()
        self.variable_map = {}
        for col in df.columns:
            col = str(col)
            variable = td.ContinuousVariable(col)
            self.variables.add(variable)
            self.variable_map[col] = variable

    # camelCase is java convention; mathcing that...
    def setParameters(self, parameters):
        self.paramaters = parameters

    @JOverride
    def localScore(self, *args):
        Xi = args[0] 
        if len(args) == 1: PAi = []
        elif isinstance(args[1], int): PAi = [args[1]]
        else: PAi = list(args[1])

        key = (Xi, *sorted(PAi))
        if key not in self.cache:
            self.cache[key] = self.score(self.data, Xi, PAi, self.parameters)
 
        # for debugging...
        # print(key, self.cache[key])
        return self.cache[key]

    @JOverride
    def localScoreDiff(self, *args):
        Xi = args[0]
        if len(args) == 2: PAi = []
        else: PAi = list(args[2])

        diff = -self.localScore(Xi, PAi)
        PAi.append(args[1])
        diff += self.localScore(Xi, PAi)

        return diff 
    
    @JOverride
    def getVariables(self):
        return self.variables
    
    @JOverride
    def getSampleSize(self):
        return self.n

    @JOverride
    def toString(self):
        return "Biwei's General Score"

    @JOverride
    def getVariable(self, targetName):
        if targetName in self.variable_map: 
            return self.variable_map[targetName]
        return None

    @JOverride
    def isEffectEdge(self, bump):
        return False

    @JOverride
    def getMaxDegree(self):
        return 1000

    @JOverride
    def defaultScore(self):
        return self
    
def dagma_nonlinear_local_learn(data):
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

def ges_non_param_local_learn(data):
    result = ges(data.values, maxP=10, node_names=data.columns, score_func="local_score_CV_general")
    return result

def ges_discrete_poisson_local_learn(data):
    result = ges(data.values, maxP=10, node_names=data.columns, score_func="local_score_BDeu")
    return result

def pc_kci_local_learn(data):
    result = pc(data.values, alpha=0.05, indep_test='kci')
    return result

def dag_gnn_local_learn(data):
    d = data.shape[1]
    X = data.values
    model = DAG_GNN(device_type="gpu")
    model.learn(X)
    return model.causal_matrix
    
def notears_mlp_local_learn(data):
    d = data.shape[1]
    X = data.values
    model = NotearsNonlinear(device_type="gpu")
    adj = model.learn(X)
    return adj

def pc_kci_tetrad_local_learn(data):
    data = data.astype({col: "float64" for col in data.columns})
    search = py_ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_kci(alpha=0.05)
    search.run_pc()
    adj = search.get_graph_to_matrix().values
    return adj

def pc_fisherz_tetrad_local_learn(data):
    data = data.astype({col: "float64" for col in data.columns})
    search = py_ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_fisher_z(alpha=0.05)
    ## Run various algorithms and print their results. For now (for compability with R)
    search.run_pc()
    adj = search.get_graph_to_matrix().values
    return adj

def ges_non_param_tetrad_local_learn(data):
    data = data.astype({col: "float64" for col in data.columns})
    score = Bgs(data)
    adj = ts.Fges(score).search()
    return adj

def ges_bic_tetrad_local_learn(data):
    data = data.astype({col: "float64" for col in data.columns})
    search = py_ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_sem_bic()
    search.run_fges()
    adj = search.get_graph_to_matrix().values
    return adj

def genie3_local_learn(data):
    adj = GENIE3(data.values, nthreads=16)
    return adj
    