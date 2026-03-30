from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass(frozen=True)
class MarkovNetworkConfig:
    alpha_grid: tuple[float, ...] = field(default_factory=lambda: tuple(np.logspace(-2, 0, 11)))
    edge_threshold: float = 1e-6 
    max_iter: int = 1000
    tolerance: float = 1e-4
    worker_tolerance: float = 1e-3  # Relaxed tolerance for shuffles
    cv_folds: int = 5
    standardize: bool = True
    adaptive_fallback: bool = True 
    verbose: bool = False

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

@dataclass(frozen=True)
class MarkovNetworkResult:
    edge_table: pd.DataFrame
    lambda_value: float
    precision_matrix: np.ndarray
    partial_correlation_matrix: np.ndarray
    variable_names: list[str]
    n_samples: int
    n_features: int

    @property
    def statistics(self) -> pd.DataFrame:
        """Returns the full edge list for all non-zero connections."""
        return self.edge_table

    @property
    def outcome_edge_table(self) -> pd.DataFrame:
        """Returns only edges directly connected to the outcome variable Y."""
        return self.edge_table[self.edge_table["Is_To_Y"]]

@dataclass(frozen=True)
class MarkovNetworkPermutationResult:
    edge_statistics: pd.DataFrame
    n_permutations: int
    alpha: float

    @property
    def significant_outcome_edges(self) -> pd.DataFrame:
        """Returns only the outcome-connected edges that passed the permutation p-value threshold."""
        return self.edge_statistics[
            (self.edge_statistics["Is_To_Y"]) & (self.edge_statistics["Is_Significant"])
        ]

class MarkovNetworkEstimator:
    def __init__(self, config: MarkovNetworkConfig | None = None) -> None:
        self.config = config or MarkovNetworkConfig()

    def fit(self, features: Any, outcome: Any, feature_names: Sequence[str] = None,
            covariates: Any = None, covariate_names: Sequence[str] = None,
            outcome_name: str = "Y") -> MarkovNetworkResult:
        """High-level wrapper to prepare data and fit the model in one call."""
        prep = self.prepare_input(features, outcome, feature_names, covariates, covariate_names, outcome_name)
        return self.fit_prepared(prep)

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
        
        # Comprehensive Input Validation
        self._validate_inputs(predictor_matrix, o_vec, variable_names)
        
        return PreparedMarkovInput(predictor_matrix.astype(float), o_vec.astype(float),
                                   predictor_names, variable_names, node_types, outcome_name)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, names: list[str]):
        """Checks for NaNs, Infs, and ensures numeric types before modeling."""
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaNs. Please clean your dataset before fitting.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains Infs. Please check for extreme values.")
        if len(names) != (X.shape[1] + 1):
            raise ValueError(f"Name mismatch: Expected {X.shape[1] + 1} names, got {len(names)}.")

    def fit_prepared(self, prep: PreparedMarkovInput, outcome_override: np.ndarray = None, 
                     fixed_alpha: float = None, pre_std_predictors: np.ndarray = None) -> MarkovNetworkResult:
        n_samples = prep.predictor_matrix.shape[0]
        
        if pre_std_predictors is not None and outcome_override is not None:
            # OPTIMIZED PATH: Only standardize the outcome (used in permutations)
            y_std = (outcome_override - outcome_override.mean()) / (outcome_override.std() + 1e-12)
            std_X = np.column_stack([pre_std_predictors, y_std])
        else:
            # STANDARD PATH: Standardize the entire matrix
            y = outcome_override if outcome_override is not None else prep.outcome_vector
            X = np.column_stack([prep.predictor_matrix, y])
            std_X, _, _ = self._standardize(X)
        
        # Determine tolerance based on whether this is a shuffle or main fit
        tol = self.config.worker_tolerance if fixed_alpha else self.config.tolerance
        
        # Stability Nudge: Compute empirical covariance and add a tiny ridge
        emp_cov = np.dot(std_X.T, std_X) / n_samples
        emp_cov += np.eye(emp_cov.shape[0]) * 1e-6
        
        precision, l_val = self._fit_precision(emp_cov, n_samples, fixed_alpha, tol)
        pcorr = self._get_pcorr(precision)
        
        edges = self._build_edges(prep.variable_names, prep.node_types, prep.outcome_name, precision, pcorr)
        return MarkovNetworkResult(
            edge_table=edges, lambda_value=l_val, 
            precision_matrix=precision, partial_correlation_matrix=pcorr,
            variable_names=prep.variable_names,
            n_samples=n_samples, n_features=std_X.shape[1]
        )

    def _fit_precision(self, emp_cov, n_samples, alpha, tol):
        if alpha is not None:
            # Staircase Fallback: Try the fixed alpha, bump slightly if convergence fails
            current_alpha = alpha
            for attempt in range(3):
                try:
                    m = GraphicalLasso(alpha=current_alpha, max_iter=self.config.max_iter, tol=tol).fit(emp_cov)
                    return m.precision_, current_alpha
                except:
                    if not self.config.adaptive_fallback: raise
                    current_alpha *= 1.5
            
        # Fallback to full Cross-Validation if staircase fails or no alpha provided
        # Ensure CV folds don't exceed sample size
        cv = min(self.config.cv_folds, n_samples - 1, 5) 
        if cv < 2: cv = 2 # Minimum folds for CV
        
        m = GraphicalLassoCV(alphas=list(self.config.alpha_grid), cv=cv, max_iter=self.config.max_iter, tol=tol).fit(emp_cov)
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
        s = StandardScaler().fit(X)
        z = s.transform(X)
        # Handle constant features (StandardScaler returns 0/NaN for zero variance)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z, s.mean_, s.scale_

    def _coerce(self, d, n, p):
        if isinstance(d, pd.DataFrame): return d.to_numpy(), list(d.columns)
        # Ensure 1D arrays are treated as a single feature with N observations (column vector)
        arr = np.array(d, dtype=float)
        if arr.ndim == 1: arr = arr[:, np.newaxis]
        m = np.atleast_2d(arr)
        return m, list(n) if n else [f"{p}_{i}" for i in range(m.shape[1])]

    def _coerce_vec(self, d):
        return np.ravel(d.to_numpy() if hasattr(d, "to_numpy") else np.array(d))

class MarkovNetworkPermutationTester:
    def __init__(self, estimator, config=None):
        self.estimator, self.config = estimator, config or PermutationTestConfig()

    def run(self, features: Any, outcome: Any, feature_names: Sequence[str] = None,
            covariates: Any = None, covariate_names: Sequence[str] = None,
            outcome_name: str = "Y") -> MarkovNetworkPermutationResult:
        """High-level wrapper to run permutation testing in one call."""
        prep = self.estimator.prepare_input(features, outcome, feature_names, covariates, covariate_names, outcome_name)
        return self.run_prepared(prep)

    def run_prepared(self, prep: PreparedMarkovInput, obs_res: MarkovNetworkResult = None) -> MarkovNetworkPermutationResult:
        obs = obs_res or self.estimator.fit_prepared(prep)
        seeds = np.random.default_rng(self.config.random_state).integers(0, 2**32, self.config.n_permutations)
        
        # Pre-standardize predictors to speed up all worker threads
        std_predictors, _, _ = self.estimator._standardize(prep.predictor_matrix)
        
        # Parallel Execution with tqdm progress monitor
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._worker)(prep, obs.lambda_value, s, std_predictors) 
            for s in tqdm(seeds, desc="Permutations", disable=not self.config.report_every)
        )

        null_dist, success = {}, 0
        for edge_list in results:
            if edge_list is None: continue
            success += 1
            for (a, b, val) in edge_list:
                key = tuple(sorted((a, b)))
                null_dist.setdefault(key, []).append(val)

        stats = []
        for r in obs.edge_table.itertuples():
            null_vals = np.array(null_dist.get(tuple(sorted((r.Node_A, r.Node_B))), []))
            # Pad null distribution with zeros if edge wasn't found in a specific shuffle
            full_null = np.concatenate([null_vals, np.zeros(success - len(null_vals))])
            p = (np.sum(full_null >= abs(r.Partial_Correlation)) + 1) / (success + 1)
            stats.append({**r._asdict(), "P_Value": p, "Is_Significant": p <= self.config.alpha})
            
        df_stats = pd.DataFrame(stats)
        if not df_stats.empty and 'Index' in df_stats.columns:
            df_stats = df_stats.drop(columns=['Index'])
            
        return MarkovNetworkPermutationResult(
            edge_statistics=df_stats,
            n_permutations=self.config.n_permutations,
            alpha=self.config.alpha
        )

    def _worker(self, prep, alpha, seed, std_predictors):
        try:
            shuffled = np.random.default_rng(seed).permutation(prep.outcome_vector)
            res = self.estimator.fit_prepared(prep, outcome_override=shuffled, 
                                             fixed_alpha=alpha, pre_std_predictors=std_predictors)
            return [(r.Node_A, r.Node_B, abs(r.Partial_Correlation)) for r in res.edge_table.itertuples()]
        except: return None

class MarkovNetworkCSVWriter:
    """Utility class to export modeling results to organized CSV files."""
    def write_fit(self, result: MarkovNetworkResult, output_dir: str | Path):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        result.edge_table.to_csv(path / "full_edge_table.csv", index=False)
        pd.DataFrame({"Lambda": [result.lambda_value]}).to_csv(path / "fit_parameters.csv", index=False)
        np.savetxt(path / "precision_matrix.csv", result.precision_matrix, delimiter=",")
        np.savetxt(path / "partial_correlation_matrix.csv", result.partial_correlation_matrix, delimiter=",")

    def write_permutation_test(self, result: MarkovNetworkPermutationResult, output_dir: str | Path):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        result.edge_statistics.to_csv(path / "permutation_statistics_all.csv", index=False)
        result.significant_outcome_edges.to_csv(path / "significant_outcome_edges.csv", index=False)

if __name__ == "__main__":
    # Robustness Test: 45 subjects, 90 parcels (N < P)
    print("Initializing Markov Network Robustness Test (N=45, P=90)...")
    rng = np.random.default_rng(42)
    X_synth = rng.normal(size=(45, 90))
    y_synth = 0.5 * X_synth[:, 0] + rng.normal(size=45)
    
    est = MarkovNetworkEstimator(MarkovNetworkConfig(verbose=True))
    tester = MarkovNetworkPermutationTester(est, PermutationTestConfig(n_permutations=20))
    
    print("Fitting Model...")
    res = est.fit(X_synth, y_synth)
    print(f"Fit complete. Lambda: {res.lambda_value:.4f}, Edges: {len(res.edge_table)}")
    
    print("Running Permutations...")
    p_res = tester.run(X_synth, y_synth)
    print(f"Success. Significant edges: {len(p_res.significant_outcome_edges)}")
