from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.preprocessing import StandardScaler


DEFAULT_ALPHA_GRID = tuple(float(value) for value in np.logspace(-4, 0, 10))


@dataclass(frozen=True)
class MarkovNetworkConfig:
    """Configuration for Graphical Lasso network fitting."""

    alpha_grid: tuple[float, ...] = field(default_factory=lambda: DEFAULT_ALPHA_GRID)
    edge_threshold: float = 0.0
    max_iter: int = 1000
    tolerance: float = 1e-4
    cv_folds: int = 5
    standardize: bool = True
    fallback_alpha: float | None = None
    verbose: bool = True


@dataclass(frozen=True)
class PermutationTestConfig:
    """Configuration for outcome permutation testing."""

    n_permutations: int = 1000
    alpha: float = 0.05
    random_state: int | None = None
    report_every: int = 50


@dataclass(frozen=True)
class PreparedMarkovInput:
    """Validated inputs ready for repeated model fitting."""

    predictor_matrix: np.ndarray
    outcome_vector: np.ndarray
    predictor_names: list[str]
    variable_names: list[str]
    node_types: dict[str, str]
    outcome_name: str


@dataclass
class NetworkFitResult:
    """Outputs from a fitted Markov network."""

    design_matrix: np.ndarray
    standardized_matrix: np.ndarray
    variable_names: list[str]
    predictor_names: list[str]
    node_types: dict[str, str]
    outcome_name: str
    covariance_matrix: np.ndarray
    correlation_matrix: np.ndarray
    precision_matrix: np.ndarray
    partial_correlation_matrix: np.ndarray
    edge_table: pd.DataFrame
    outcome_edge_table: pd.DataFrame
    node_summary: pd.DataFrame
    statistics: pd.DataFrame
    lambda_value: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def design_matrix_frame(self) -> pd.DataFrame:
        """Return the unscaled design matrix as a DataFrame."""

        return pd.DataFrame(self.design_matrix, columns=self.variable_names)

    def standardized_matrix_frame(self) -> pd.DataFrame:
        """Return the standardized design matrix as a DataFrame."""

        return pd.DataFrame(self.standardized_matrix, columns=self.variable_names)

    def covariance_frame(self) -> pd.DataFrame:
        """Return the covariance matrix as a DataFrame."""

        return pd.DataFrame(
            self.covariance_matrix,
            index=self.variable_names,
            columns=self.variable_names,
        )

    def correlation_frame(self) -> pd.DataFrame:
        """Return the correlation matrix as a DataFrame."""

        return pd.DataFrame(
            self.correlation_matrix,
            index=self.variable_names,
            columns=self.variable_names,
        )

    def precision_frame(self) -> pd.DataFrame:
        """Return the precision matrix as a DataFrame."""

        return pd.DataFrame(
            self.precision_matrix,
            index=self.variable_names,
            columns=self.variable_names,
        )

    def partial_correlation_frame(self) -> pd.DataFrame:
        """Return the partial correlation matrix as a DataFrame."""

        return pd.DataFrame(
            self.partial_correlation_matrix,
            index=self.variable_names,
            columns=self.variable_names,
        )

    def scaler_frame(self) -> pd.DataFrame:
        """Return scaler parameters for each variable."""

        return pd.DataFrame(
            {
                "Variable": self.variable_names,
                "Mean": self.scaler_mean,
                "Scale": self.scaler_scale,
            }
        )


@dataclass
class PermutationTestResult:
    """Outputs from permutation testing."""

    observed_result: NetworkFitResult
    edge_statistics: pd.DataFrame
    significant_edges: pd.DataFrame
    significant_outcome_edges: pd.DataFrame
    filtered_node_summary: pd.DataFrame
    filtered_statistics: pd.DataFrame
    filtered_partial_correlation_matrix: np.ndarray
    successful_permutations: int
    alpha: float

    def filtered_partial_correlation_frame(self) -> pd.DataFrame:
        """Return the permutation-filtered partial correlation matrix."""

        names = self.observed_result.variable_names
        return pd.DataFrame(
            self.filtered_partial_correlation_matrix,
            index=names,
            columns=names,
        )


class MarkovNetworkEstimator:
    """Fit sparse Markov networks from array-like inputs."""

    def __init__(self, config: MarkovNetworkConfig | None = None) -> None:
        self.config = config or MarkovNetworkConfig()

    def prepare_input(
        self,
        features: Any,
        outcome: Any,
        feature_names: Sequence[str] | None = None,
        covariates: Any | None = None,
        covariate_names: Sequence[str] | None = None,
        outcome_name: str = "Y",
    ) -> PreparedMarkovInput:
        """Validate arrays and prepare a reusable design specification."""

        feature_matrix, resolved_feature_names = self._coerce_matrix(
            data=features,
            names=feature_names,
            default_prefix="Feature",
        )
        outcome_vector = self._coerce_vector(outcome, name=outcome_name)

        if feature_matrix.shape[0] != outcome_vector.shape[0]:
            raise ValueError("Features and outcome must have the same number of rows.")

        predictor_matrix = feature_matrix
        predictor_names = list(resolved_feature_names)
        node_types = {name: "feature" for name in predictor_names}

        if covariates is not None:
            covariate_matrix, resolved_covariate_names = self._coerce_matrix(
                data=covariates,
                names=covariate_names,
                default_prefix="Covariate",
            )
            if covariate_matrix.shape[0] != outcome_vector.shape[0]:
                raise ValueError("Covariates and outcome must have the same number of rows.")
            predictor_matrix = np.column_stack([predictor_matrix, covariate_matrix])
            predictor_names.extend(resolved_covariate_names)
            node_types.update({name: "covariate" for name in resolved_covariate_names})

        if outcome_name in predictor_names:
            raise ValueError(f"Outcome name '{outcome_name}' also appears in predictor names.")

        self._validate_finite(predictor_matrix, "predictor matrix")
        self._validate_finite(outcome_vector, "outcome vector")

        variable_names = predictor_names + [outcome_name]
        node_types[outcome_name] = "outcome"

        return PreparedMarkovInput(
            predictor_matrix=predictor_matrix.astype(float, copy=False),
            outcome_vector=outcome_vector.astype(float, copy=False),
            predictor_names=predictor_names,
            variable_names=variable_names,
            node_types=node_types,
            outcome_name=outcome_name,
        )

    def fit(
        self,
        features: Any,
        outcome: Any,
        feature_names: Sequence[str] | None = None,
        covariates: Any | None = None,
        covariate_names: Sequence[str] | None = None,
        outcome_name: str = "Y",
    ) -> NetworkFitResult:
        """Fit a Markov network from features, optional covariates, and outcome."""

        prepared_input = self.prepare_input(
            features=features,
            outcome=outcome,
            feature_names=feature_names,
            covariates=covariates,
            covariate_names=covariate_names,
            outcome_name=outcome_name,
        )
        return self.fit_prepared(prepared_input)

    def fit_prepared(
        self,
        prepared_input: PreparedMarkovInput,
        outcome_override: np.ndarray | None = None,
    ) -> NetworkFitResult:
        """Fit a Markov network from prepared inputs."""

        outcome_vector = prepared_input.outcome_vector
        if outcome_override is not None:
            outcome_vector = self._coerce_vector(
                outcome_override,
                name=prepared_input.outcome_name,
            )
            if outcome_vector.shape[0] != prepared_input.predictor_matrix.shape[0]:
                raise ValueError("Outcome override must match the predictor row count.")
            self._validate_finite(outcome_vector, "outcome override")

        design_matrix = np.column_stack([prepared_input.predictor_matrix, outcome_vector])
        standardized_matrix, scaler_mean, scaler_scale = self._standardize(design_matrix)
        covariance_matrix = self._covariance_matrix(standardized_matrix)
        correlation_matrix = self._correlation_matrix(covariance_matrix)
        precision_matrix, lambda_value = self._fit_precision_model(standardized_matrix)
        partial_correlation_matrix = self._partial_correlation_matrix(precision_matrix)

        edge_table = self._build_edge_table(
            variable_names=prepared_input.variable_names,
            node_types=prepared_input.node_types,
            outcome_name=prepared_input.outcome_name,
            precision_matrix=precision_matrix,
            partial_correlation_matrix=partial_correlation_matrix,
        )
        outcome_edge_table = self._build_outcome_edge_table(
            edge_table=edge_table,
            outcome_name=prepared_input.outcome_name,
        )
        node_summary = self._build_node_summary(
            variable_names=prepared_input.variable_names,
            node_types=prepared_input.node_types,
            edge_table=edge_table,
            outcome_name=prepared_input.outcome_name,
            partial_correlation_matrix=partial_correlation_matrix,
        )
        statistics = self._build_statistics(
            n_samples=design_matrix.shape[0],
            variable_names=prepared_input.variable_names,
            outcome_edge_table=outcome_edge_table,
            edge_table=edge_table,
            lambda_value=lambda_value,
        )

        return NetworkFitResult(
            design_matrix=design_matrix,
            standardized_matrix=standardized_matrix,
            variable_names=prepared_input.variable_names,
            predictor_names=prepared_input.predictor_names,
            node_types=prepared_input.node_types,
            outcome_name=prepared_input.outcome_name,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            precision_matrix=precision_matrix,
            partial_correlation_matrix=partial_correlation_matrix,
            edge_table=edge_table,
            outcome_edge_table=outcome_edge_table,
            node_summary=node_summary,
            statistics=statistics,
            lambda_value=lambda_value,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

    def fit_design_matrix(
        self,
        design_matrix: Any,
        variable_names: Sequence[str] | None = None,
        outcome_index: int = -1,
    ) -> NetworkFitResult:
        """Fit a Markov network when the outcome is already embedded in the matrix."""

        matrix, resolved_names = self._coerce_matrix(
            data=design_matrix,
            names=variable_names,
            default_prefix="Variable",
        )
        resolved_index = outcome_index if outcome_index >= 0 else matrix.shape[1] + outcome_index
        if resolved_index < 0 or resolved_index >= matrix.shape[1]:
            raise ValueError("Outcome index is out of bounds for the provided matrix.")

        outcome_name = resolved_names[resolved_index]
        feature_mask = np.ones(matrix.shape[1], dtype=bool)
        feature_mask[resolved_index] = False
        predictor_matrix = matrix[:, feature_mask]
        predictor_names = [name for i, name in enumerate(resolved_names) if i != resolved_index]

        prepared_input = PreparedMarkovInput(
            predictor_matrix=predictor_matrix,
            outcome_vector=matrix[:, resolved_index],
            predictor_names=predictor_names,
            variable_names=predictor_names + [outcome_name],
            node_types={**{name: "feature" for name in predictor_names}, outcome_name: "outcome"},
            outcome_name=outcome_name,
        )
        return self.fit_prepared(prepared_input)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)

    def _fit_precision_model(self, standardized_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        alpha_grid = np.asarray(self.config.alpha_grid, dtype=float)
        if alpha_grid.ndim != 1 or alpha_grid.size == 0:
            raise ValueError("alpha_grid must contain at least one value.")

        cv_folds = min(self.config.cv_folds, standardized_matrix.shape[0] - 1)
        if alpha_grid.size > 1 and cv_folds >= 2:
            try:
                self._log("Fitting GraphicalLassoCV.")
                model = GraphicalLassoCV(
                    alphas=alpha_grid,
                    cv=cv_folds,
                    max_iter=self.config.max_iter,
                    tol=self.config.tolerance,
                    enet_tol=1e-3,
                )
                model.fit(standardized_matrix)
                return model.precision_, float(model.alpha_)
            except Exception as error:
                self._log(f"GraphicalLassoCV failed: {error}")

        candidate_alphas = self._fallback_alphas(alpha_grid)
        last_error: Exception | None = None
        for alpha in candidate_alphas:
            try:
                self._log(f"Fitting GraphicalLasso with alpha={alpha:.6f}.")
                model = GraphicalLasso(
                    alpha=alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.tolerance,
                )
                model.fit(standardized_matrix)
                return model.precision_, float(alpha)
            except Exception as error:
                last_error = error
                self._log(f"GraphicalLasso failed for alpha={alpha:.6f}: {error}")

        raise RuntimeError("Unable to fit a Graphical Lasso model.") from last_error

    def _fallback_alphas(self, alpha_grid: np.ndarray) -> list[float]:
        fallback_values: list[float] = []
        if self.config.fallback_alpha is not None:
            fallback_values.append(float(self.config.fallback_alpha))
        fallback_values.extend(sorted({float(value) for value in alpha_grid}, reverse=True))
        return fallback_values

    def _standardize(self, design_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.standardize:
            scaler = StandardScaler()
            standardized_matrix = scaler.fit_transform(design_matrix)
            scale = np.where(np.asarray(scaler.scale_, dtype=float) == 0, 1.0, scaler.scale_)
            return standardized_matrix, np.asarray(scaler.mean_, dtype=float), scale

        center = np.mean(design_matrix, axis=0)
        centered_matrix = design_matrix - center
        scale = np.ones(design_matrix.shape[1], dtype=float)
        return centered_matrix, center, scale

    @staticmethod
    def _covariance_matrix(matrix: np.ndarray) -> np.ndarray:
        centered_matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
        return centered_matrix.T @ centered_matrix / matrix.shape[0]

    @staticmethod
    def _correlation_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
        scale = np.sqrt(np.clip(np.diag(covariance_matrix), a_min=0.0, a_max=None))
        denominator = np.outer(scale, scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_matrix = np.divide(
                covariance_matrix,
                denominator,
                out=np.zeros_like(covariance_matrix),
                where=denominator > 0,
            )
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix

    @staticmethod
    def _partial_correlation_matrix(precision_matrix: np.ndarray) -> np.ndarray:
        diagonal = np.sqrt(np.clip(np.diag(precision_matrix), a_min=0.0, a_max=None))
        denominator = np.outer(diagonal, diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            partial = np.divide(
                -precision_matrix,
                denominator,
                out=np.zeros_like(precision_matrix),
                where=denominator > 0,
            )
        np.fill_diagonal(partial, 1.0)
        return np.clip(partial, -1.0, 1.0)

    def _build_edge_table(
        self,
        variable_names: Sequence[str],
        node_types: dict[str, str],
        outcome_name: str,
        precision_matrix: np.ndarray,
        partial_correlation_matrix: np.ndarray,
    ) -> pd.DataFrame:
        edge_records: list[dict[str, Any]] = []
        n_variables = len(variable_names)

        for row_index in range(n_variables):
            node_a = variable_names[row_index]
            for column_index in range(row_index + 1, n_variables):
                node_b = variable_names[column_index]
                precision_value = float(precision_matrix[row_index, column_index])
                if abs(precision_value) <= self.config.edge_threshold:
                    continue

                partial_correlation = float(partial_correlation_matrix[row_index, column_index])
                edge_records.append(
                    {
                        "Node_A": node_a,
                        "Node_B": node_b,
                        "Node_A_Type": node_types[node_a],
                        "Node_B_Type": node_types[node_b],
                        "Partial_Correlation": partial_correlation,
                        "Abs_Partial_Correlation": abs(partial_correlation),
                        "Precision": precision_value,
                        "Abs_Precision": abs(precision_value),
                        "Is_To_Y": outcome_name in {node_a, node_b},
                    }
                )

        edge_table = pd.DataFrame(edge_records)
        if edge_table.empty:
            return pd.DataFrame(
                columns=[
                    "Node_A",
                    "Node_B",
                    "Node_A_Type",
                    "Node_B_Type",
                    "Partial_Correlation",
                    "Abs_Partial_Correlation",
                    "Precision",
                    "Abs_Precision",
                    "Is_To_Y",
                ]
            )
        return edge_table.sort_values(
            by=["Is_To_Y", "Abs_Partial_Correlation"],
            ascending=[False, False],
        ).reset_index(drop=True)

    @staticmethod
    def _build_outcome_edge_table(edge_table: pd.DataFrame, outcome_name: str) -> pd.DataFrame:
        if edge_table.empty:
            return pd.DataFrame(
                columns=[
                    "Node",
                    "Node_Type",
                    "Partial_Correlation",
                    "Abs_Partial_Correlation",
                    "Precision",
                    "Abs_Precision",
                ]
            )

        outcome_edges = edge_table[edge_table["Is_To_Y"]].copy()
        outcome_edges["Node"] = np.where(
            outcome_edges["Node_A"] == outcome_name,
            outcome_edges["Node_B"],
            outcome_edges["Node_A"],
        )
        outcome_edges["Node_Type"] = np.where(
            outcome_edges["Node_A"] == outcome_name,
            outcome_edges["Node_B_Type"],
            outcome_edges["Node_A_Type"],
        )
        return outcome_edges[
            [
                "Node",
                "Node_Type",
                "Partial_Correlation",
                "Abs_Partial_Correlation",
                "Precision",
                "Abs_Precision",
            ]
        ].reset_index(drop=True)

    @staticmethod
    def _build_node_summary(
        variable_names: Sequence[str],
        node_types: dict[str, str],
        edge_table: pd.DataFrame,
        outcome_name: str,
        partial_correlation_matrix: np.ndarray,
    ) -> pd.DataFrame:
        degree_by_node = {name: 0 for name in variable_names}
        outcome_neighbors: set[str] = set()

        if not edge_table.empty:
            for row in edge_table.itertuples(index=False):
                degree_by_node[row.Node_A] += 1
                degree_by_node[row.Node_B] += 1
                if row.Is_To_Y:
                    outcome_neighbors.add(row.Node_A)
                    outcome_neighbors.add(row.Node_B)

        outcome_index = list(variable_names).index(outcome_name)
        node_records: list[dict[str, Any]] = []
        for index, node_name in enumerate(variable_names):
            node_records.append(
                {
                    "Node": node_name,
                    "Node_Type": node_types[node_name],
                    "Degree": degree_by_node[node_name],
                    "Connected_To_Y": node_name in outcome_neighbors,
                    "Outcome_Partial_Correlation": float(
                        partial_correlation_matrix[index, outcome_index]
                    ),
                }
            )

        return pd.DataFrame(node_records).sort_values(
            by=["Connected_To_Y", "Degree", "Node"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    def _build_statistics(
        self,
        n_samples: int,
        variable_names: Sequence[str],
        outcome_edge_table: pd.DataFrame,
        edge_table: pd.DataFrame,
        lambda_value: float,
    ) -> pd.DataFrame:
        n_variables = len(variable_names)
        possible_edges = n_variables * (n_variables - 1) / 2
        density = float(len(edge_table) / possible_edges) if possible_edges else 0.0
        mean_degree = float((2 * len(edge_table)) / n_variables) if n_variables else 0.0

        return pd.DataFrame(
            [
                {
                    "N_Samples": n_samples,
                    "N_Variables": n_variables,
                    "N_Edges": len(edge_table),
                    "N_Outcome_Edges": len(outcome_edge_table),
                    "Density": density,
                    "Mean_Degree": mean_degree,
                    "Selected_Alpha": lambda_value,
                    "Edge_Threshold": self.config.edge_threshold,
                }
            ]
        )

    @staticmethod
    def _coerce_matrix(
        data: Any,
        names: Sequence[str] | None,
        default_prefix: str,
    ) -> tuple[np.ndarray, list[str]]:
        if isinstance(data, pd.DataFrame):
            matrix = data.to_numpy(dtype=float)
            resolved_names = list(data.columns) if names is None else list(names)
        else:
            matrix = np.asarray(data, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            if matrix.ndim != 2:
                raise ValueError("Input data must be 2-dimensional.")
            resolved_names = (
                list(names)
                if names is not None
                else [f"{default_prefix}_{index + 1}" for index in range(matrix.shape[1])]
            )

        if len(resolved_names) != matrix.shape[1]:
            raise ValueError("The number of provided names does not match the column count.")
        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("Variable names must be unique.")
        return matrix.astype(float, copy=False), resolved_names

    @staticmethod
    def _coerce_vector(data: Any, name: str) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            vector = data.iloc[:, 0].to_numpy(dtype=float)
        elif isinstance(data, pd.Series):
            vector = data.to_numpy(dtype=float)
        else:
            vector = np.asarray(data, dtype=float)

        vector = np.ravel(vector)
        if vector.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        return vector.astype(float, copy=False)

    @staticmethod
    def _validate_finite(data: np.ndarray, label: str) -> None:
        if not np.isfinite(data).all():
            raise ValueError(f"{label} contains NaN or infinite values.")


class MarkovNetworkPermutationTester:
    """Run permutation testing for a fitted Markov network."""

    def __init__(
        self,
        estimator: MarkovNetworkEstimator,
        config: PermutationTestConfig | None = None,
    ) -> None:
        self.estimator = estimator
        self.config = config or PermutationTestConfig()

    def run(
        self,
        features: Any,
        outcome: Any,
        feature_names: Sequence[str] | None = None,
        covariates: Any | None = None,
        covariate_names: Sequence[str] | None = None,
        outcome_name: str = "Y",
    ) -> PermutationTestResult:
        """Run permutation testing from raw arrays."""

        prepared_input = self.estimator.prepare_input(
            features=features,
            outcome=outcome,
            feature_names=feature_names,
            covariates=covariates,
            covariate_names=covariate_names,
            outcome_name=outcome_name,
        )
        return self.run_prepared(prepared_input)

    def run_prepared(
        self,
        prepared_input: PreparedMarkovInput,
        observed_result: NetworkFitResult | None = None,
    ) -> PermutationTestResult:
        """Run permutation testing from prepared inputs."""

        observed_result = observed_result or self.estimator.fit_prepared(prepared_input)
        random_generator = np.random.default_rng(self.config.random_state)
        null_distributions: dict[tuple[str, str], list[float]] = {}
        successful_permutations = 0

        for permutation_index in range(self.config.n_permutations):
            shuffled_outcome = random_generator.permutation(prepared_input.outcome_vector)
            try:
                permuted_result = self.estimator.fit_prepared(
                    prepared_input,
                    outcome_override=shuffled_outcome,
                )
            except Exception as error:
                self.estimator._log(
                    f"Permutation {permutation_index + 1} failed and was skipped: {error}"
                )
                continue

            successful_permutations += 1
            for row in permuted_result.edge_table.itertuples(index=False):
                edge_key = self._edge_key(row.Node_A, row.Node_B)
                null_distributions.setdefault(edge_key, []).append(
                    abs(float(row.Partial_Correlation))
                )

            should_report = (
                self.config.report_every > 0
                and (permutation_index + 1) % self.config.report_every == 0
            )
            if should_report:
                self.estimator._log(
                    f"Completed {permutation_index + 1} permutations "
                    f"({successful_permutations} successful)."
                )

        if successful_permutations == 0:
            raise RuntimeError("All permutations failed; no null distribution was created.")

        edge_statistics = self._build_edge_statistics(
            observed_result=observed_result,
            null_distributions=null_distributions,
            successful_permutations=successful_permutations,
        )
        significant_edges = edge_statistics[edge_statistics["Is_Significant"]].copy()
        significant_outcome_edges = significant_edges[significant_edges["Is_To_Y"]].copy()
        filtered_matrix = self._filtered_partial_correlation_matrix(
            observed_result=observed_result,
            significant_edges=significant_edges,
        )
        filtered_node_summary = self.estimator._build_node_summary(
            variable_names=observed_result.variable_names,
            node_types=observed_result.node_types,
            edge_table=self._edge_statistics_to_edge_table(significant_edges),
            outcome_name=observed_result.outcome_name,
            partial_correlation_matrix=filtered_matrix,
        )
        filtered_statistics = self.estimator._build_statistics(
            n_samples=observed_result.design_matrix.shape[0],
            variable_names=observed_result.variable_names,
            outcome_edge_table=self._significant_outcome_edge_table(significant_outcome_edges),
            edge_table=self._edge_statistics_to_edge_table(significant_edges),
            lambda_value=observed_result.lambda_value,
        )

        return PermutationTestResult(
            observed_result=observed_result,
            edge_statistics=edge_statistics,
            significant_edges=self._edge_statistics_to_edge_table(significant_edges),
            significant_outcome_edges=self._significant_outcome_edge_table(
                significant_outcome_edges
            ),
            filtered_node_summary=filtered_node_summary,
            filtered_statistics=filtered_statistics,
            filtered_partial_correlation_matrix=filtered_matrix,
            successful_permutations=successful_permutations,
            alpha=self.config.alpha,
        )

    def _build_edge_statistics(
        self,
        observed_result: NetworkFitResult,
        null_distributions: dict[tuple[str, str], list[float]],
        successful_permutations: int,
    ) -> pd.DataFrame:
        edge_statistics: list[dict[str, Any]] = []
        for row in observed_result.edge_table.itertuples(index=False):
            edge_key = self._edge_key(row.Node_A, row.Node_B)
            null_abs_values = np.asarray(null_distributions.get(edge_key, []), dtype=float)
            observed_abs_value = abs(float(row.Partial_Correlation))

            if null_abs_values.size == 0:
                p_value = 1.0 / (successful_permutations + 1)
                null_mean = np.nan
                null_std = np.nan
            else:
                p_value = float(
                    (np.sum(null_abs_values >= observed_abs_value) + 1)
                    / (null_abs_values.size + 1)
                )
                null_mean = float(np.mean(null_abs_values))
                null_std = float(np.std(null_abs_values))

            edge_statistics.append(
                {
                    "Node_A": row.Node_A,
                    "Node_B": row.Node_B,
                    "Node_A_Type": row.Node_A_Type,
                    "Node_B_Type": row.Node_B_Type,
                    "Partial_Correlation": float(row.Partial_Correlation),
                    "Abs_Partial_Correlation": observed_abs_value,
                    "Precision": float(row.Precision),
                    "Abs_Precision": float(row.Abs_Precision),
                    "Null_Mean_Abs_Partial_Correlation": null_mean,
                    "Null_Std_Abs_Partial_Correlation": null_std,
                    "Permutation_Count": int(null_abs_values.size),
                    "P_Value": p_value,
                    "Is_Significant": p_value <= self.config.alpha,
                    "Is_To_Y": bool(row.Is_To_Y),
                }
            )

        edge_statistics_frame = pd.DataFrame(edge_statistics)
        if edge_statistics_frame.empty:
            return pd.DataFrame(
                columns=[
                    "Node_A",
                    "Node_B",
                    "Node_A_Type",
                    "Node_B_Type",
                    "Partial_Correlation",
                    "Abs_Partial_Correlation",
                    "Precision",
                    "Abs_Precision",
                    "Null_Mean_Abs_Partial_Correlation",
                    "Null_Std_Abs_Partial_Correlation",
                    "Permutation_Count",
                    "P_Value",
                    "Is_Significant",
                    "Is_To_Y",
                ]
            )
        return edge_statistics_frame.sort_values(
            by=["P_Value", "Abs_Partial_Correlation"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def _filtered_partial_correlation_matrix(
        self,
        observed_result: NetworkFitResult,
        significant_edges: pd.DataFrame,
    ) -> np.ndarray:
        filtered_matrix = np.eye(len(observed_result.variable_names), dtype=float)
        name_to_index = {
            name: index for index, name in enumerate(observed_result.variable_names)
        }

        for row in significant_edges.itertuples(index=False):
            index_a = name_to_index[row.Node_A]
            index_b = name_to_index[row.Node_B]
            filtered_matrix[index_a, index_b] = float(row.Partial_Correlation)
            filtered_matrix[index_b, index_a] = float(row.Partial_Correlation)

        return filtered_matrix

    @staticmethod
    def _edge_statistics_to_edge_table(edge_statistics: pd.DataFrame) -> pd.DataFrame:
        if edge_statistics.empty:
            return pd.DataFrame(
                columns=[
                    "Node_A",
                    "Node_B",
                    "Node_A_Type",
                    "Node_B_Type",
                    "Partial_Correlation",
                    "Abs_Partial_Correlation",
                    "Precision",
                    "Abs_Precision",
                    "Is_To_Y",
                ]
            )
        return edge_statistics[
            [
                "Node_A",
                "Node_B",
                "Node_A_Type",
                "Node_B_Type",
                "Partial_Correlation",
                "Abs_Partial_Correlation",
                "Precision",
                "Abs_Precision",
                "Is_To_Y",
            ]
        ].reset_index(drop=True)

    @staticmethod
    def _significant_outcome_edge_table(significant_outcome_edges: pd.DataFrame) -> pd.DataFrame:
        if significant_outcome_edges.empty:
            return pd.DataFrame(
                columns=[
                    "Node",
                    "Node_Type",
                    "Partial_Correlation",
                    "Abs_Partial_Correlation",
                    "Precision",
                    "Abs_Precision",
                    "P_Value",
                ]
            )

        outcome_edges = significant_outcome_edges.copy()
        outcome_edges["Node"] = np.where(
            outcome_edges["Is_To_Y"] & (outcome_edges["Node_A_Type"] == "outcome"),
            outcome_edges["Node_B"],
            outcome_edges["Node_A"],
        )
        outcome_edges["Node_Type"] = np.where(
            outcome_edges["Is_To_Y"] & (outcome_edges["Node_A_Type"] == "outcome"),
            outcome_edges["Node_B_Type"],
            outcome_edges["Node_A_Type"],
        )
        return outcome_edges[
            [
                "Node",
                "Node_Type",
                "Partial_Correlation",
                "Abs_Partial_Correlation",
                "Precision",
                "Abs_Precision",
                "P_Value",
            ]
        ].reset_index(drop=True)

    @staticmethod
    def _edge_key(node_a: str, node_b: str) -> tuple[str, str]:
        return tuple(sorted((node_a, node_b)))


class MarkovNetworkCSVWriter:
    """Write model outputs to CSV files."""

    def write_fit(
        self,
        fit_result: NetworkFitResult,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write unfiltered network outputs."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {
            "subject_level_features_unscaled.csv": fit_result.design_matrix_frame(),
            "subject_level_features_scaled.csv": fit_result.standardized_matrix_frame(),
            "graph_edges.csv": fit_result.edge_table,
            "graph_y_connections.csv": fit_result.outcome_edge_table,
            "graph_node_summary.csv": fit_result.node_summary,
            "graph_partial_correlation_matrix.csv": fit_result.partial_correlation_frame(),
            "graph_precision_matrix.csv": fit_result.precision_frame(),
            "graph_simple_correlation_matrix.csv": fit_result.correlation_frame(),
            "graph_covariance_matrix.csv": fit_result.covariance_frame(),
            "graph_statistics.csv": fit_result.statistics,
            "scaler_parameters.csv": fit_result.scaler_frame(),
        }
        return self._write_frames(output_files, output_path)

    def write_permutation_test(
        self,
        permutation_result: PermutationTestResult,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write permutation outputs and filtered network summaries."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {
            "permutation_analysis_results.csv": permutation_result.edge_statistics,
            "graph_filtered_edges.csv": permutation_result.significant_edges,
            "graph_filtered_y_connections.csv": permutation_result.significant_outcome_edges,
            "graph_filtered_node_summary.csv": permutation_result.filtered_node_summary,
            "graph_filtered_partial_correlation_matrix.csv": (
                permutation_result.filtered_partial_correlation_frame()
            ),
            "graph_filtered_statistics.csv": permutation_result.filtered_statistics,
        }
        return self._write_frames(output_files, output_path)

    @staticmethod
    def _write_frames(
        output_files: dict[str, pd.DataFrame],
        output_path: Path,
    ) -> dict[str, Path]:
        written_files: dict[str, Path] = {}
        for filename, frame in output_files.items():
            file_path = output_path / filename
            frame.to_csv(file_path, index=False if frame.index.equals(pd.RangeIndex(len(frame))) else True)
            written_files[filename] = file_path
        return written_files


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Fit a Markov network from a CSV file and optionally run permutation testing."
    )
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--outcome-column",
        required=True,
        help="Name of the outcome column in the input CSV.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        help="Feature columns to include. Defaults to every non-outcome, non-covariate column.",
    )
    parser.add_argument(
        "--covariate-columns",
        nargs="*",
        default=[],
        help="Optional covariate columns to include alongside the main features.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV outputs should be written.",
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="*",
        type=float,
        default=list(DEFAULT_ALPHA_GRID),
        help="Graphical Lasso alpha grid.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute precision value required to keep an edge.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for the Graphical Lasso solver.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Convergence tolerance for the Graphical Lasso solver.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum number of cross-validation folds for GraphicalLassoCV.",
    )
    parser.add_argument(
        "--fallback-alpha",
        type=float,
        default=None,
        help="Fallback alpha to try before values from the alpha grid.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Center the data without scaling it to unit variance.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Number of permutations to run. Set to 0 to skip permutation testing.",
    )
    parser.add_argument(
        "--permutation-alpha",
        type=float,
        default=0.05,
        help="Significance threshold for permutation testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for permutation testing.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Progress reporting interval for permutation testing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    data_frame = pd.read_csv(args.input_csv)
    covariate_columns = list(args.covariate_columns)
    if args.outcome_column not in data_frame.columns:
        raise ValueError(f"Outcome column '{args.outcome_column}' was not found.")

    if args.feature_columns:
        feature_columns = list(args.feature_columns)
    else:
        excluded_columns = {args.outcome_column, *covariate_columns}
        feature_columns = [column for column in data_frame.columns if column not in excluded_columns]

    missing_columns = set(feature_columns + covariate_columns + [args.outcome_column]) - set(
        data_frame.columns
    )
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The following columns were not found in the input CSV: {missing_text}")

    feature_frame = data_frame[feature_columns]
    covariate_frame = data_frame[covariate_columns] if covariate_columns else None
    outcome_series = data_frame[args.outcome_column]

    estimator = MarkovNetworkEstimator(
        MarkovNetworkConfig(
            alpha_grid=tuple(float(value) for value in args.alpha_grid),
            edge_threshold=float(args.edge_threshold),
            max_iter=int(args.max_iter),
            tolerance=float(args.tolerance),
            cv_folds=int(args.cv_folds),
            standardize=not args.no_standardize,
            fallback_alpha=args.fallback_alpha,
            verbose=not args.quiet,
        )
    )
    prepared_input = estimator.prepare_input(
        features=feature_frame,
        outcome=outcome_series,
        feature_names=feature_columns,
        covariates=covariate_frame,
        covariate_names=covariate_columns,
        outcome_name=args.outcome_column,
    )
    fit_result = estimator.fit_prepared(prepared_input)

    writer = MarkovNetworkCSVWriter()
    writer.write_fit(fit_result, args.output_dir)

    if args.n_permutations > 0:
        permutation_tester = MarkovNetworkPermutationTester(
            estimator=estimator,
            config=PermutationTestConfig(
                n_permutations=int(args.n_permutations),
                alpha=float(args.permutation_alpha),
                random_state=args.random_state,
                report_every=int(args.report_every),
            ),
        )
        permutation_result = permutation_tester.run_prepared(
            prepared_input=prepared_input,
            observed_result=fit_result,
        )
        writer.write_permutation_test(permutation_result, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
