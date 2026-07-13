### Step 1: Data Loading and Initial Feature Set Definition

First, we need to get our data ready.
*   **1.1 Load Data:** Load the `wdbc.csv` dataset into a pandas DataFrame.
*   **1.2 Separate Features and Target:** Isolate the 30 feature columns (X) from the `target` column (y). Remember, `target` is encoded as 0 = malignant, 1 = benign.
*   **1.3 Identify `Worst` Features:** Based on the data description, we know that `worst` features are typically more discriminative. For this project, we will initially restrict our analysis to only the `worst` features. This means selecting the 10 columns that end with "worst" (e.g., `worst radius`, `worst texture`, etc.) from our feature set X. This will be our primary feature matrix for subsequent steps.

### Step 2: Define Feature Groups for Multicollinearity Management

The data description highlights strong multicollinearity. To address this systematically, especially with regularization, we need to explicitly define these groups among our `worst` features.
*   **2.1 Group `Radius`/`Perimeter`/`Area`:** Identify `worst radius`, `worst perimeter`, and `worst area` as one highly correlated group.
*   **2.2 Group `Compactness`/`Concavity`/`Concave Points`:** Identify `worst compactness`, `worst concavity`, and `worst concave points` as another highly correlated group.
*   **2.3 Remaining Features:** The other `worst` features (`worst texture`, `worst smoothness`, `worst symmetry`, `worst fractal dimension`) will be treated as individual features, as their inter-correlations are less extreme or not explicitly highlighted as problematic for group-wise selection. This grouping will inform our interpretation of selected features.

### Step 3: Implement Nested Cross-Validation Framework

Given our small sample size (569 samples), a single train/test split is insufficient for honest performance estimation. We will use nested cross-validation.
*   **3.1 Outer Loop:** Set up a K-fold cross-validation (e.g., K=5 or K=10) for model evaluation. Use `StratifiedKFold` to preserve the proportion of target classes in each fold. This loop will provide an unbiased estimate of the model's performance on unseen data.
*   **3.2 Inner Loop:** Within each fold of the outer loop, set up another K'-fold cross-validation (e.g., K'=5) for hyperparameter tuning. Use `StratifiedKFold` for the inner loop as well. This ensures that hyperparameter selection does not leak information from the outer test set.
*   **3.3 Data Splitting:** For each outer fold, the data will be split into an outer training set and an outer test set. The outer training set will then be further split into inner training and validation sets for hyperparameter tuning.

### Step 4: Preprocessing within Cross-Validation Folds

Preprocessing steps, especially standardization, must be applied *within* each cross-validation fold to prevent data leakage.
*   **4.1 Standardization:** For each inner training fold, fit a `StandardScaler` to the features. Then, transform the inner training, inner validation, and outer test sets using this *fitted* scaler. This ensures that scaling parameters are learned only from the training data.
*   **4.2 Feature Set Consistency:** Ensure that only the `worst` features, as defined in Step 1.3, are used throughout the scaling and modeling process.

### Step 5: Model Training with Regularized Feature Selection

We will use a Logistic Regression model with L1 regularization (Lasso) to perform feature selection and manage multicollinearity among the `worst` features.
*   **5.1 Model Choice:** Employ Logistic Regression, which is well-suited for binary classification and known to perform well on this dataset.
*   **5.2 L1 Regularization (Lasso):** Apply L1 regularization. Lasso inherently performs feature selection by driving some feature coefficients to zero. Crucially, when features are highly correlated (as in our defined groups), Lasso tends to select only one representative feature from that group, effectively addressing our goal of identifying minimal, representative features and acting as a form of group-aware selection.
*   **5.3 Class Weighting:** To account for the mild class imbalance and prioritize the minority (malignant) class, configure the Logistic Regression model to use `class_weight='balanced'`.
*   **5.4 Hyperparameter Tuning (Inner Loop):** Within each inner cross-validation loop, we will tune the regularization strength (e.g., the `C` parameter for Logistic Regression, which is the inverse of regularization strength `alpha`). The tuning will aim to optimize a robust metric like the Area Under the Receiver Operating Characteristic curve (AUC-ROC) on the inner validation sets.

### Step 6: Model Evaluation and Metric Calculation

After hyperparameter tuning in the inner loop, the best model will be evaluated on the outer test set.
*   **6.1 Train Best Model:** For each outer fold, train the Logistic Regression model with the optimal `C` parameter (found in the inner loop) and `class_weight='balanced'` on the entire outer training set.
*   **6.2 Predict Probabilities:** Generate predicted probabilities for the malignant class (target=0) on the outer test set.
*   **6.3 Calculate Clinical Metrics:**
    *   **Sensitivity at High Specificity:** For each outer test set, we will determine a decision threshold that achieves a specificity of at least 95%. This threshold will be identified by analyzing the Receiver Operating Characteristic (ROC) curve, selecting the highest probability threshold that results in a specificity of 95% or greater. We will then report the sensitivity (recall of malignant cases) at this specific threshold.
    *   **Probability Calibration:** Assess the calibration of predicted probabilities using metrics like the Brier score and by generating reliability diagrams.
*   **6.4 Store Results:** Store the true labels, predicted probabilities, and calculated metrics (sensitivity, specificity, Brier score, and the chosen threshold) for each outer fold.

### Step 7: Aggregate Results and Compute Confidence Intervals

Once all outer folds are processed, we will aggregate the results to get a robust overall performance estimate.
*   **7.1 Aggregate Metrics:** Average the sensitivity, specificity, and Brier scores across all outer folds.
*   **7.2 Compute Confidence Intervals:** Calculate 95% confidence intervals for all aggregated metrics (sensitivity, specificity, Brier score) using appropriate statistical methods (e.g., bootstrapping or standard error of the mean across folds). This is crucial due to the small sample size.

### Step 8: Feature Interpretation and Diagnostic Signature Characterization

The final step involves interpreting the selected features and their impact.
*   **8.1 Aggregate Feature Coefficients:** Collect the coefficients of the `worst` features from the best models trained in each outer fold.
*   **8.2 Identify Consistently Selected Features:** Analyze which `worst` features consistently have non-zero coefficients across the outer folds. Quantify this consistency by reporting the frequency or percentage of outer folds in which each feature's coefficient was non-zero. Pay particular attention to which feature was selected from the highly correlated groups (e.g., `worst radius`, `worst perimeter`, `worst area`).
*   **8.3 Characterize Diagnostic Signatures:** Based on the consistently selected and highly weighted `worst` features, describe the distinct morphological patterns that drive malignancy. For instance, if `worst radius` is consistently selected over `worst perimeter` and `worst area`, we will interpret this as `worst radius` being the primary representative morphological driver for that group. This will allow us to move beyond general feature importance to pinpoint core morphological patterns.