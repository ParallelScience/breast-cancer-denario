### Step 1: Data Loading and Initial Inspection

The first step involves loading the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
1.  **Load Data:** Read the `wdbc.csv` file into a suitable data structure.
2.  **Separate Features and Target:** Identify the 30 feature columns and the `target` column. The `target` column (0 = malignant, 1 = benign) will be our dependent variable, and the 30 nuclear morphology measurements will be our independent variables.
3.  **Verify Data Integrity:** Confirm the absence of missing values and ensure all feature columns are numerical, as indicated in the data description.

### Step 2: Feature Engineering - Derived Distributional Characteristics

This crucial step involves deriving new features that represent key characteristics of the "latent intra-tumor morphological distributions" for each of the ten base nuclear morphology measurements. Given that we only have three summary statistics (`mean`, `error`, and `worst`) per base measurement, directly fitting complex parametric or semi-parametric distributions is challenging. Therefore, we will derive pragmatic, interpretable proxies that capture essential distributional characteristics such as asymmetry, relative variability, and tail extremity.

1.  **Identify Base Measurement Groups:** Group the 30 original features into 10 sets, each corresponding to a base measurement and its three statistics (e.g., `mean radius`, `radius error`, `worst radius`).
2.  **Clarify `error_X` Interpretation:** The data description states `error` is "standard error over nuclei" (`std_dev / sqrt(N)`). For the purpose of these derived features, we will treat `error_X` as a measure of relative spread, acknowledging it is `std_dev / sqrt(N)`. We assume that `N` (number of nuclei) is either constant across samples or that its variability does not invalidate the relative comparisons of spread captured by these proxies. This is a pragmatic assumption given the available data.
3.  **Derive Skewness Proxy:** For each base measurement `X`, calculate a `skewness_proxy_X` feature using the formula: `(worst_X - mean_X) / error_X`. This metric quantifies the relative deviation of the upper tail (`worst`) from the central tendency (`mean`), scaled by the spread (`error`), thereby providing an indication of the distribution's asymmetry.
4.  **Derive Coefficient of Variation Proxy:** For each base measurement `X`, calculate a `cv_proxy_X` feature using the formula: `error_X / mean_X`. This represents the relative variability of the measurement within the tumor, approximating the coefficient of variation. Care will be taken to handle cases where `mean_X` might be close to zero, though the data description states features are strictly positive.
5.  **Derive Tail-to-Mean Ratio:** For each base measurement `X`, calculate a `tail_ratio_X` feature using the formula: `worst_X / mean_X`. This feature captures the relative magnitude of the extreme values compared to the average, highlighting the severity of the upper tail.
6.  **Construct New Feature Set:** These calculations will generate 30 new features (3 for each of the 10 base measurements). These will form our "derived features" set.

### Step 3: Feature Set Definition

To rigorously evaluate the impact of the newly derived features, we will define three distinct feature sets for model training and evaluation:

1.  **Feature Set A (Original Features):** Comprises the initial 30 nuclear morphology measurements (`mean radius`, `texture error`, `worst perimeter`, etc.).
2.  **Feature Set B (Derived Features):** Consists of the 30 newly engineered features (e.g., `skewness_proxy_radius`, `cv_proxy_texture`, `tail_ratio_perimeter`).
3.  **Feature Set C (Combined Features):** An concatenation of Feature Set A and Feature Set B, resulting in a total of 60 features.

### Step 4: Data Preprocessing

Prior to model training, all feature sets will undergo standardization to ensure features are on a comparable scale, which is crucial for many machine learning algorithms and explicitly mentioned as important in the data description.

1.  **Standardization:** Apply `StandardScaler` to each feature set (A, B, C). This will transform the features to have a mean of 0 and a standard deviation of 1. It is critical that this scaling is performed *within* each cross-validation fold (i.e., fitted only on the training data of a fold) to prevent data leakage.

### Step 5: Model Selection and Nested Cross-Validation Setup

Given the small sample size (569 samples) and the need for robust performance estimates, a nested cross-validation strategy will be employed.

1.  **Model Selection:** Select appropriate classification algorithms. We will include models known for good performance on linearly separable or near-linearly separable data, such as Logistic Regression and Support Vector Machines (with a linear kernel), which offer some interpretability. Additionally, to explore potential non-linear patterns in the derived features, a non-linear model like Random Forest will be included.
2.  **Nested Cross-Validation:** Implement a nested cross-validation scheme.
    *   **Outer Loop:** An outer 5-fold cross-validation will be used for unbiased performance estimation. Each fold will provide a test set for final evaluation.
    *   **Inner Loop:** An inner 3-fold cross-validation will be used within each outer training fold for hyperparameter tuning of the selected models. This ensures that hyperparameter optimization does not bias the final performance metrics.

### Step 6: Model Training and Hyperparameter Tuning

For each chosen model and each feature set, the following steps will be executed within the nested cross-validation framework:

1.  **Hyperparameter Tuning:** Within the inner cross-validation loop, perform a grid search or randomized search to find the optimal hyperparameters for the model (e.g., regularization strength `C` for Logistic Regression and SVM, or tree parameters for Random Forest). The optimization criterion for the inner loop will prioritize `roc_auc` or a custom scoring function designed to maximize sensitivity at a high specificity, directly aligning with the primary clinical evaluation goals.
2.  **Model Training:** Using the best hyperparameters identified from the inner loop, train the model on the full training data of the current outer cross-validation fold.

### Step 7: Performance Evaluation and Confidence Interval Estimation

Model performance will be rigorously evaluated on the held-out test set of each outer cross-validation fold, focusing on clinically relevant metrics.

1.  **Prediction:** Generate predicted probabilities and class labels for the test set of each outer fold.
2.  **Metric Calculation:** Calculate the following metrics for each fold:
    *   **Sensitivity (Malignant Recall) at High Specificity:** Determine the sensitivity (true positive rate for malignant class) when specificity (true negative rate for benign class) is set at clinically relevant thresholds (e.g., 95% and 98%). For each outer cross-validation fold, the probability threshold corresponding to the desired specificity will be determined from the predicted probabilities on the *training data* of that fold (or, more robustly, on the validation sets from the inner cross-validation loop). This determined threshold will then be applied to the held-out test set of the outer fold to calculate the sensitivity. This approach directly addresses the clinical priority of minimizing false negatives, reflecting the higher cost of a missed cancer diagnosis.
    *   **Area Under the Receiver Operating Characteristic Curve (AUROC):** A robust measure of overall discriminative power.
    *   **Brier Score:** To assess the calibration of predicted probabilities.
3.  **Confidence Intervals:** Compute the mean and confidence intervals (e.g., 95% confidence intervals) for all reported metrics across the outer cross-validation folds. This will provide a robust estimate of model performance and its uncertainty.

### Step 8: Comparative Analysis and Interpretation

The final step involves comparing the performance across the different feature sets and drawing conclusions regarding the utility of the derived features.

1.  **Performance Comparison:** Systematically compare the mean performance metrics and their confidence intervals for models trained on Feature Set A (Original), Feature Set B (Derived), and Feature Set C (Combined). This will directly test the hypothesis that the derived features provide superior discrimination.
2.  **Feature Contribution Analysis and Multicollinearity Assessment:** Analyze the relative contribution of the original versus the derived features. For linear models, examine the magnitude and sign of the learned coefficients. To explicitly address the strong multicollinearity among the original features (and potentially between original and derived features), we will:
    *   Examine the correlation matrix between the original and derived features to understand their interdependencies.
    *   Consider reporting Variance Inflation Factors (VIFs) for the derived features to quantify their collinearity with other features.
    *   Acknowledge that strong multicollinearity can make direct per-feature interpretation challenging and affect the stability of model coefficients, even for regularized models. Therefore, the primary focus will be on the overall discriminative power added by the *group* of derived features rather than precise individual feature weights. This will inform our interpretation of their unique contribution.
\