from __future__ import annotations

import warnings
import multiprocessing
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass(frozen=True)
class MarkovNetworkConfig:
    alpha_grid: tuple[float, ...] = field(default_factory=lambda: tuple(np.logspace(-4, 0, 10)))
    edge_threshold: float = 1e-6 
    max_iter: int = 1000
    tolerance: float = 1e-4
    cv_folds: int = 5
    standardize: bool = True
    adaptive_fallback: bool = True 

@dataclass(frozen=True)
class PermutationTestConfig:
    n_permutations: int = 1000
    alpha: float = 0.05
    random_state: int | None = None
    n_jobs: int = -1  # Set to -1 for max speed, or 1 to 4 if running other apps
    report_every: int = 50

@dataclass(frozen=True)
class PreparedMarkovInput:
    predictor_matrix: np.ndarray
    outcome_vector: np.ndarray
    predictor_names: list[str]
    variable_names: list[str]
    node_types: dict[str, str]
    outcome_name: str

class MarkovNetworkEstimator:
    def __init__(self, config: MarkovNetworkConfig | None = None) -> None:
        self.config = config or MarkovNetworkConfig()

    def prepare_input(self, features: Any, outcome: Any, feature_names: Sequence[str] = None,
                      covariates: Any = None, covariate_names: Sequence[str] = None,
                      outcome_name: str = "Y") -> PreparedMarkovInput:
        f_mat, f_names = self._coerce(features, feature_names, "Feature")
        o_vec = self._coerce_vec(outcome)
        predictor_matrix, predictor_names = f_mat, list(f_names)
        node_types = {name: "feature" for name in predictor_names}
        
        if covariates is not None:
            c_mat, c_names = self._coerce(covariates, covariate_names, "Covariate")
            predictor_matrix = np.column_stack([predictor_matrix, c_mat])
            predictor_names.extend(c_names)
            node_types.update({name: "covariate" for name in c_names})
            
        variable_names = predictor_names + [outcome_name]
        node_types[outcome_name] = "outcome"
        return PreparedMarkovInput(predictor_matrix.astype(float), o_vec.astype(float),
                                   predictor_names, variable_names, node_types, outcome_name)

    def fit_prepared(self, prep: PreparedMarkovInput, outcome_override: np.ndarray = None, 
                     fixed_alpha: float = None) -> Any:
        y = outcome_override if outcome_override is not None else prep.outcome_vector
        X = np.column_stack([prep.predictor_matrix, y])
        std_X, _, _ = self._standardize(X)
        
        # Adaptive Logic: Try fixed alpha, fallback to CV if it fails
        precision, l_val = self._fit_precision(std_X, fixed_alpha)
        pcorr = self._get_pcorr(precision)
        
        # Vectorized edge discovery
        edges = self._build_edges(prep.variable_names, prep.node_types, prep.outcome_name, precision, pcorr)
        
        # Return a simple dictionary for workers, or a full result for the main fit
        return type('Result', (), {'edge_table': edges, 'lambda_value': l_val, 'precision': precision, 'pcorr': pcorr})

    def _fit_precision(self, std_X, alpha):
        if alpha is not None:
            try:
                m = GraphicalLasso(alpha=alpha, max_iter=self.config.max_iter, tol=self.config.tolerance).fit(std_X)
                return m.precision_, alpha
            except:
                if not self.config.adaptive_fallback: raise
        cv = min(self.config.cv_folds, std_X.shape[0] - 1)
        m = GraphicalLassoCV(alphas=list(self.config.alpha_grid), cv=cv, max_iter=self.config.max_iter).fit(std_X)
        return m.precision_, float(m.alpha_)

    def _build_edges(self, names, types, y_name, prec, pcorr):
        n = len(names); iu = np.triu_indices(n, k=1)
        mask = np.abs(prec[iu]) > self.config.edge_threshold
        if not np.any(mask): return pd.DataFrame()
        
        a_idx, b_idx = iu[0][mask], iu[1][mask]
        name_arr, type_arr = np.array(names), np.array([types[n] for n in names])
        return pd.DataFrame({
            "Node_A": name_arr[a_idx], "Node_B": name_arr[b_idx],
            "Node_A_Type": type_arr[a_idx], "Node_B_Type": type_arr[b_idx],
            "Partial_Correlation": pcorr[iu][mask], "Is_To_Y": (name_arr[a_idx] == y_name) | (name_arr[b_idx] == y_name)
        })

    def _get_pcorr(self, prec):
        d = np.sqrt(np.diag(prec)); outer = np.outer(d, d)
        p = np.divide(-prec, outer, out=np.zeros_like(prec), where=outer > 0)
        np.fill_diagonal(p, 1.0); return np.clip(p, -1.0, 1.0)

    def _standardize(self, X):
        if not self.config.standardize: return X - X.mean(0), X.mean(0), np.ones(X.shape[1])
        s = StandardScaler().fit(X); return s.transform(X), s.mean_, s.scale_

    def _coerce(self, d, n, p):
        if isinstance(d, pd.DataFrame): return d.to_numpy(), list(d.columns)
        m = np.atleast_2d(np.array(d, dtype=float)); return m, list(n) if n else [f"{p}_{i}" for i in range(m.shape[1])]

    def _coerce_vec(self, d):
        return np.ravel(d.to_numpy() if hasattr(d, "to_numpy") else np.array(d))

class MarkovNetworkPermutationTester:
    def __init__(self, estimator, config=None):
        self.estimator, self.config = estimator, config or PermutationTestConfig()

    def run_prepared(self, prep, obs_res=None):
        obs = obs_res or self.estimator.fit_prepared(prep)
        seeds = np.random.default_rng(self.config.random_state).integers(0, 2**32, self.config.n_permutations)
        
        # Parallel Execution with Thread Control (n_jobs)
        results = Parallel(n_jobs=self.config.n_jobs, verbose=10)(
            delayed(self._worker)(prep, obs.lambda_value, s) for s in seeds
        )

        null_dist, success = {}, 0
        for edge_list in results:
            if edge_list is None: continue
            success += 1
            for (a, b, val) in edge_list:
                key = tuple(sorted((a, b))); null_dist.setdefault(key, []).append(val)

        stats = []
        for r in obs.edge_table.itertuples():
            null_vals = np.array(null_dist.get(tuple(sorted((r.Node_A, r.Node_B))), []))
            p = (np.sum(null_vals >= abs(r.Partial_Correlation)) + 1) / (success + 1)
            stats.append({**r._asdict(), "P_Value": p, "Is_Significant": p <= self.config.alpha})
        return pd.DataFrame(stats)

    def _worker(self, prep, alpha, seed):
        try:
            shuffled = np.random.default_rng(seed).permutation(prep.outcome_vector)
            res = self.estimator.fit_prepared(prep, outcome_override=shuffled, fixed_alpha=alpha)
            return [(r.Node_A, r.Node_B, abs(r.Partial_Correlation)) for r in res.edge_table.itertuples()]
        except: return None
