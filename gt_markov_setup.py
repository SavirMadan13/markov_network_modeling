from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DEFAULT_NETWORK_NAMES = {
    1: "VIS",
    2: "SM",
    3: "DA",
    4: "VA",
    5: "LIM",
    6: "FP",
    7: "DMN",
}


@dataclass(frozen=True)
class GTColumnConfig:
    efield_column: str = "efield_file"
    atrophy_column: str = "atrophy_data_v2"
    connectivity_column: str = "t_file"
    age_column: str | None = "age"
    # If provided, this column is used directly as the outcome and baseline/followup are ignored.
    outcome_column: str | None = None
    baseline_outcome_column: str | None = "Baseline ADAS-Cog11"
    followup_outcome_column: str | None = "12 months Post-Stimulation ADAS-Cog 11"
    global_atrophy_column: str | None = None

    def resolved_global_atrophy_column(self) -> str:
        return self.global_atrophy_column or self.atrophy_column


@dataclass(frozen=True)
class GTSubsetConfig:
    enabled: bool = False
    method: str = "global_atrophy"
    fraction: float = 0.5
    selection: str = "top"
    n_subjects: int | None = None
    random_state: int = 42


@dataclass(frozen=True)
class GTFeatureConfig:
    include_age: bool = False
    include_atrophy: bool = True
    include_fc: bool = True
    include_interactions: bool = True


@dataclass(frozen=True)
class LoadedParcellation:
    voxel_labels: np.ndarray
    network_names_dict: dict[int, str]
    network_names: list[str]


@dataclass(frozen=True)
class PreparedGTMarkovData:
    subject_table: pd.DataFrame
    feature_table: pd.DataFrame
    features: pd.DataFrame
    covariates: pd.DataFrame | None
    outcome: pd.Series
    network_feature_records: list[dict[str, dict[str, float]]]
    parcellation: LoadedParcellation
    outcome_name: str


OutcomeBuilder = Callable[[pd.Series, GTColumnConfig], float]


def load_subject_table(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_path).expanduser())


def load_parcellation(
    parcellation_path: str | Path,
    network_name_overrides: Mapping[int, str] | None = None,
) -> LoadedParcellation:
    parcellation_img = nib.load(str(Path(parcellation_path).expanduser()))
    voxel_labels = np.asarray(parcellation_img.get_fdata(), dtype=float).ravel()

    network_labels = np.unique(voxel_labels)
    network_labels = network_labels[network_labels > 0]
    if network_labels.size == 0:
        raise ValueError("Parcellation does not contain any positive network labels.")

    default_names = (
        DEFAULT_NETWORK_NAMES
        if network_labels.size <= len(DEFAULT_NETWORK_NAMES)
        else {int(label): f"Network_{int(label)}" for label in network_labels}
    )
    merged_names = {**default_names, **{int(k): v for k, v in (network_name_overrides or {}).items()}}
    network_names_dict = {
        int(label): merged_names.get(int(label), f"Network_{int(label)}")
        for label in sorted(network_labels)
    }
    return LoadedParcellation(
        voxel_labels=voxel_labels,
        network_names_dict=network_names_dict,
        network_names=list(network_names_dict.values()),
    )


def load_subject_modalities(
    row: pd.Series,
    parcellation: np.ndarray,
    columns: GTColumnConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = columns or GTColumnConfig()
    efield = _load_flat_image(row, columns.efield_column)
    atrophy = _load_flat_image(row, columns.atrophy_column)
    connectivity = _load_flat_image(row, columns.connectivity_column)

    if not (efield.size == atrophy.size == connectivity.size == parcellation.size):
        raise ValueError("Subject imaging data and parcellation must have the same number of voxels.")

    mask = (
        np.isfinite(efield)
        & np.isfinite(atrophy)
        & np.isfinite(connectivity)
        & np.isfinite(parcellation)
    )
    if not np.any(mask):
        raise ValueError("No valid voxels remain after masking NaNs.")
    return efield[mask], atrophy[mask], connectivity[mask], parcellation[mask]


def calculate_global_atrophy(
    df: pd.DataFrame,
    columns: GTColumnConfig | None = None,
) -> tuple[np.ndarray, list[object]]:
    columns = columns or GTColumnConfig()
    global_atrophy: list[float] = []
    valid_indices: list[object] = []

    for idx, row in df.iterrows():
        try:
            atrophy = _load_flat_image(row, columns.resolved_global_atrophy_column())
        except Exception:
            continue

        if not np.isfinite(atrophy).any():
            continue
        global_atrophy.append(float(np.nanmean(atrophy)))
        valid_indices.append(idx)

    return np.asarray(global_atrophy, dtype=float), valid_indices


def filter_dataframe_subset(
    df: pd.DataFrame,
    subset_config: GTSubsetConfig | None = None,
    columns: GTColumnConfig | None = None,
) -> pd.DataFrame:
    subset_config = subset_config or GTSubsetConfig()
    columns = columns or GTColumnConfig()
    if not subset_config.enabled:
        return df.copy()
    if df.empty:
        return df.copy()

    n_select = _resolve_subset_size(len(df), subset_config)

    if subset_config.method == "global_atrophy":
        global_atrophy, valid_indices = calculate_global_atrophy(df, columns=columns)
        if len(valid_indices) == 0:
            raise ValueError("Could not compute global atrophy for any subject.")

        n_select = _resolve_subset_size(len(valid_indices), subset_config)
        sorted_indices = np.argsort(global_atrophy)

        if subset_config.selection == "top":
            selected_local_indices = sorted_indices[-n_select:]
        elif subset_config.selection == "bottom":
            selected_local_indices = sorted_indices[:n_select]
        elif subset_config.selection == "middle":
            start_idx = max(0, len(sorted_indices) // 2 - n_select // 2)
            end_idx = start_idx + n_select
            selected_local_indices = sorted_indices[start_idx:end_idx]
        else:
            raise ValueError(f"Unknown subset selection: {subset_config.selection}")

        selected_df_indices = [valid_indices[i] for i in selected_local_indices]
        return df.loc[selected_df_indices].copy()

    if subset_config.method == "random":
        return df.sample(n=n_select, random_state=subset_config.random_state).copy()
    if subset_config.method == "first_n":
        return df.iloc[:n_select].copy()
    if subset_config.method == "last_n":
        return df.iloc[-n_select:].copy()
    raise ValueError(f"Unknown subset method: {subset_config.method}")


def compute_outcome(row: pd.Series, columns: GTColumnConfig | None = None) -> float:
    columns = columns or GTColumnConfig()
    if columns.outcome_column is not None:
        value = row.get(columns.outcome_column, np.nan)
        return float(value) if pd.notna(value) else np.nan

    if not columns.baseline_outcome_column or not columns.followup_outcome_column:
        raise ValueError(
            "Set outcome_column to a precomputed improvement column, or provide both "
            "baseline_outcome_column and followup_outcome_column."
        )

    baseline = row.get(columns.baseline_outcome_column, np.nan)
    followup = row.get(columns.followup_outcome_column, np.nan)
    if pd.isna(baseline) or pd.isna(followup) or baseline == 0:
        return np.nan
    return float((baseline - followup) / baseline)


def extract_network_features(
    df: pd.DataFrame,
    parcellation: LoadedParcellation,
    columns: GTColumnConfig | None = None,
    include_age: bool = False,
    outcome_builder: OutcomeBuilder | None = None,
) -> tuple[list[dict[str, dict[str, float]]], np.ndarray, np.ndarray | None, pd.Index]:
    columns = columns or GTColumnConfig()
    outcome_builder = outcome_builder or compute_outcome

    network_features: list[dict[str, dict[str, float]]] = []
    outcomes: list[float] = []
    ages: list[float] = []
    kept_indices: list[object] = []

    for idx, row in df.iterrows():
        try:
            _, atrophy, connectivity, parc_masked = load_subject_modalities(
                row,
                parcellation.voxel_labels,
                columns=columns,
            )
        except Exception:
            continue

        subject_networks: dict[str, dict[str, float]] = {}
        for net_label, net_name in parcellation.network_names_dict.items():
            network_mask = parc_masked == net_label
            if not np.any(network_mask):
                subject_networks[net_name] = {"A_mean": 0.0, "FC_mean": 0.0}
                continue

            subject_networks[net_name] = {
                "A_mean": float(np.mean(atrophy[network_mask])),
                "FC_mean": float(np.mean(connectivity[network_mask])),
            }

        outcome_value = outcome_builder(row, columns)
        age_value = np.nan
        if include_age:
            if columns.age_column is None:
                raise ValueError("include_age=True but no age column was provided.")
            age_value = row.get(columns.age_column, np.nan)

        if not _is_valid_subject(subject_networks, outcome_value, age_value if include_age else None):
            continue

        network_features.append(subject_networks)
        outcomes.append(float(outcome_value))
        if include_age:
            ages.append(float(age_value))
        kept_indices.append(idx)

    age_array = np.asarray(ages, dtype=float) if include_age else None
    return (
        network_features,
        np.asarray(outcomes, dtype=float),
        age_array,
        pd.Index(kept_indices),
    )


def construct_feature_matrix(
    network_features: Sequence[dict[str, dict[str, float]]],
    network_names: Sequence[str],
    outcomes: Sequence[float],
    ages: Sequence[float] | None = None,
    feature_config: GTFeatureConfig | None = None,
    outcome_name: str = "Y",
) -> pd.DataFrame:
    feature_config = feature_config or GTFeatureConfig()
    if not feature_config.include_atrophy and not feature_config.include_fc:
        raise ValueError("At least one of include_atrophy or include_fc must be True.")

    columns: dict[str, Sequence[float]] = {}

    for network_name in network_names:
        if feature_config.include_atrophy:
            columns[f"{network_name}_A"] = [
                feature_row[network_name]["A_mean"] for feature_row in network_features
            ]
        if feature_config.include_fc:
            columns[f"{network_name}_FC"] = [
                feature_row[network_name]["FC_mean"] for feature_row in network_features
            ]

    # Mirrors the active interaction logic in gt.py: A x FC only.
    if (
        feature_config.include_interactions
        and feature_config.include_atrophy
        and feature_config.include_fc
    ):
        for left_network in network_names:
            for right_network in network_names:
                columns[f"{left_network}_A_x_{right_network}_FC"] = [
                    feature_row[left_network]["A_mean"] * feature_row[right_network]["FC_mean"]
                    for feature_row in network_features
                ]

    if feature_config.include_age:
        if ages is None:
            raise ValueError("include_age=True but ages were not provided.")
        columns["Age"] = ages

    columns[outcome_name] = outcomes
    return pd.DataFrame(columns)


def split_feature_table(
    feature_table: pd.DataFrame,
    outcome_column: str = "Y",
    covariate_columns: Sequence[str] | None = ("Age",),
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
    covariate_columns = covariate_columns or ()
    available_covariates = [
        column
        for column in covariate_columns
        if column in feature_table.columns and column != outcome_column
    ]
    feature_columns = [
        column
        for column in feature_table.columns
        if column not in {*available_covariates, outcome_column}
    ]

    features = feature_table[feature_columns].copy()
    covariates = feature_table[available_covariates].copy() if available_covariates else None
    outcome = feature_table[outcome_column].copy()
    return features, covariates, outcome


def prepare_markov_network_inputs(
    dataframe: pd.DataFrame,
    parcellation_path: str | Path,
    columns: GTColumnConfig | None = None,
    feature_config: GTFeatureConfig | None = None,
    subset_config: GTSubsetConfig | None = None,
    network_name_overrides: Mapping[int, str] | None = None,
    outcome_builder: OutcomeBuilder | None = None,
    outcome_name: str = "Y",
) -> PreparedGTMarkovData:
    columns = columns or GTColumnConfig()
    feature_config = feature_config or GTFeatureConfig()
    subset_config = subset_config or GTSubsetConfig()

    parcellation = load_parcellation(
        parcellation_path,
        network_name_overrides=network_name_overrides,
    )
    subject_table = filter_dataframe_subset(
        dataframe,
        subset_config=subset_config,
        columns=columns,
    )
    network_feature_records, outcomes, ages, kept_indices = extract_network_features(
        subject_table,
        parcellation,
        columns=columns,
        include_age=feature_config.include_age,
        outcome_builder=outcome_builder,
    )

    subject_table = subject_table.loc[kept_indices].copy()
    feature_table = construct_feature_matrix(
        network_feature_records,
        parcellation.network_names,
        outcomes,
        ages=ages,
        feature_config=feature_config,
        outcome_name=outcome_name,
    )
    feature_table.index = subject_table.index

    covariate_columns = ("Age",) if feature_config.include_age else ()
    features, covariates, outcome = split_feature_table(
        feature_table,
        outcome_column=outcome_name,
        covariate_columns=covariate_columns,
    )
    outcome.name = outcome_name

    return PreparedGTMarkovData(
        subject_table=subject_table,
        feature_table=feature_table,
        features=features,
        covariates=covariates,
        outcome=outcome,
        network_feature_records=network_feature_records,
        parcellation=parcellation,
        outcome_name=outcome_name,
    )


def load_and_prepare_markov_network_inputs(
    data_path: str | Path,
    parcellation_path: str | Path,
    columns: GTColumnConfig | None = None,
    feature_config: GTFeatureConfig | None = None,
    subset_config: GTSubsetConfig | None = None,
    network_name_overrides: Mapping[int, str] | None = None,
    outcome_builder: OutcomeBuilder | None = None,
    outcome_name: str = "Y",
) -> PreparedGTMarkovData:
    dataframe = load_subject_table(data_path)
    return prepare_markov_network_inputs(
        dataframe,
        parcellation_path,
        columns=columns,
        feature_config=feature_config,
        subset_config=subset_config,
        network_name_overrides=network_name_overrides,
        outcome_builder=outcome_builder,
        outcome_name=outcome_name,
    )


def prepare_estimator_input(estimator, prepared_data: PreparedGTMarkovData):
    return estimator.prepare_input(
        features=prepared_data.features,
        outcome=prepared_data.outcome,
        covariates=prepared_data.covariates,
        outcome_name=prepared_data.outcome_name,
    )


def _load_flat_image(row: pd.Series, column_name: str) -> np.ndarray:
    value = row.get(column_name, np.nan)
    if pd.isna(value):
        raise ValueError(f"Missing image path for column {column_name!r}.")
    return np.asarray(nib.load(str(Path(value).expanduser())).get_fdata(), dtype=float).ravel()


def _resolve_subset_size(total_subjects: int, subset_config: GTSubsetConfig) -> int:
    if subset_config.n_subjects is not None:
        return min(subset_config.n_subjects, total_subjects)

    n_select = int(total_subjects * subset_config.fraction)
    if n_select < 1:
        raise ValueError("Subset configuration selected zero subjects.")
    return min(n_select, total_subjects)


def _is_valid_subject(
    subject_networks: dict[str, dict[str, float]],
    outcome_value: float,
    age_value: float | None,
) -> bool:
    values = [outcome_value]
    if age_value is not None:
        values.append(age_value)
    for network_values in subject_networks.values():
        values.extend(network_values.values())
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


__all__ = [
    "GTColumnConfig",
    "GTFeatureConfig",
    "GTSubsetConfig",
    "LoadedParcellation",
    "PreparedGTMarkovData",
    "calculate_global_atrophy",
    "compute_outcome",
    "construct_feature_matrix",
    "extract_network_features",
    "filter_dataframe_subset",
    "load_and_prepare_markov_network_inputs",
    "load_parcellation",
    "load_subject_modalities",
    "load_subject_table",
    "prepare_estimator_input",
    "prepare_markov_network_inputs",
    "split_feature_table",
]
