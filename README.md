# Markov Network Modeling (Graphical Lasso)

This repository provides a robust framework for estimating Markov Networks (Gaussian Graphical Models) using the **Graphical Lasso** algorithm, specifically optimized for high-dimensional clinical and neuroimaging data where the number of features ($P$) often exceeds the number of subjects ($N$).

## Setup & Installation

The repository includes a `requirements.txt` and is optimized for use with `uv`.

```bash
# Create and sync environment
uv venv
uv pip install -r requirements.txt
```

## Performance & Optimization

This implementation is highly optimized for large-scale permutation testing:
*   **Predictor Pre-standardization**: Predictors are standardized once in the main process, rather than 1,000 times across worker threads. This results in a **~30x speedup** on typical datasets.
*   **Vectorized Edge Discovery**: Matrix-level masking using `np.triu_indices` extracts the network graph in a single block operation.
*   **Benchmark**: On a MacBook with $N=45$ and $P=90$, the system performs at **~270+ permutations/second**.

### 1. Graphical Lasso (Glasso)
The model estimates a sparse precision matrix $\Theta = \Sigma^{-1}$ by maximizing the penalized log-likelihood:
$$\log \det \Theta - \text{tr}(S\Theta) - \alpha \|\Theta\|_1$$
where $S$ is the empirical covariance matrix and $\alpha$ is the regularization parameter.

### 2. Handling $N < P$ Challenges
When $P > N$, the empirical covariance matrix is singular, making the inversion unstable. This implementation includes several "Numerical Moats" to ensure convergence:
*   **Adaptive Alpha Grid**: The default grid starts at $10^{-2}$ rather than $10^{-4}$ to avoid non-convergent regions.
*   **Staircase Fallback**: If a fixed $\alpha$ fails to converge during permutations, the tool automatically increments $\alpha$ by 50% up to 3 times before falling back to full Cross-Validation.
*   **Relaxed Worker Tolerance**: Permutation shuffles use a tolerance of $10^{-3}$ (vs $10^{-4}$ for the main fit) to significantly speed up the null distribution generation without biasing p-values.

## Hyperparameters & Config

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `alpha_grid` | `logspace(-2, 0, 11)` | Range of penalties searched during Cross-Validation. |
| `edge_threshold` | `1e-6` | Threshold for pruning near-zero edges in the partial correlation matrix. |
| `tolerance` | `1e-4` | Convergence threshold for the main model fit. |
| `worker_tolerance`| `1e-3` | Convergence threshold for permutation shuffles (optimized for speed). |
| `standardize` | `True` | Z-scores all features and outcomes before modeling (highly recommended). |
| `n_permutations` | `1000` | Number of outcome shuffles to build the null distribution for p-values. |

## Assumptions

1.  **Multivariate Normality**: The model assumes data is Gaussian. For highly skewed data, consider a rank-based transform (non-paranormal) before input.
2.  **Sparsity**: The model assumes that the "true" network is relatively sparse (most partial correlations are zero).
3.  **Exchangeability**: Permutation testing assumes that under the null hypothesis, the outcome labels are exchangeable across subjects.

## Output Structure

The `MarkovNetworkCSVWriter` exports the following:
*   `full_edge_table.csv`: All non-zero edges in the network.
*   `permutation_statistics_all.csv`: Full statistics including p-values and significance for all edges.
*   `significant_outcome_edges.csv`: The primary output—edges connected to your outcome variable ($Y$) that passed the alpha threshold (e.g., $p < 0.05$).
*   `precision_matrix.csv`: The raw sparse precision matrix.
