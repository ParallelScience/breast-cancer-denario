# breast-cancer-denario

**Scientist:** denario
**Date:** 2026-07-13

## Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

**Source.** The UCI / scikit-learn `load_breast_cancer` dataset (Street, Wolberg & Mangasarian, 1993). A local copy is provided at `data/wdbc.csv` in this project.

**Content.** 569 samples, each derived from a digitized image of a fine-needle aspirate (FNA) of a breast mass. Features describe characteristics of the cell nuclei present in the image.

**Target.** Binary diagnosis: `malignant` (212 samples, 37.3%) vs `benign` (357 samples, 62.7%). Mildly imbalanced. Encoded in the `target` column as 0 = malignant, 1 = benign (scikit-learn convention).

**Features.** 30 real-valued predictors, all strictly positive and on very different scales (so standardization matters). Ten base nuclear-morphology measurements are each summarized by three statistics across the nuclei in an image, giving 30 columns:

Base measurements (10): `radius` (mean of distances from center to points on the perimeter), `texture` (std. dev. of gray-scale values), `perimeter`, `area`, `smoothness` (local variation in radius lengths), `compactness` (perimeter^2 / area - 1.0), `concavity` (severity of concave portions of the contour), `concave points` (number of concave portions of the contour), `symmetry`, `fractal dimension` ("coastline approximation" - 1).

Statistics (3): `mean` (mean over nuclei), `error` (standard error over nuclei), `worst` (mean of the three largest values).

Column names follow the pattern `mean radius`, `radius error`, `worst radius`, etc.

**Known structure and analysis caveats.**
- Strong multicollinearity by construction: `radius`, `perimeter` and `area` are near-deterministic functions of one another (|r| > 0.97), and `compactness`/`concavity`/`concave points` are highly correlated. Any interpretation of per-feature importance must confront this.
- The `worst` (upper-tail) features are typically more discriminative than the `mean` features — tail severity, not average appearance, carries the diagnostic signal.
- The task is close to linearly separable after standardization; well-tuned linear models reach ~97-98% accuracy, so headline accuracy is a saturated and uninformative metric. Meaningful evaluation should focus on the clinically relevant operating regime: sensitivity (malignant recall) at high specificity, calibration of predicted probabilities, and the cost asymmetry between a false negative (missed cancer) and a false positive (unnecessary biopsy).
- No missing values. No patient identifiers, no longitudinal follow-up, no survival outcomes — this is a single-timepoint diagnostic classification dataset, not a prognostic one.
- 569 samples is small: nested cross-validation (not a single train/test split) is required for any honest performance estimate, and confidence intervals on all reported metrics are essential.

**File format.** `data/wdbc.csv` — 569 rows x 31 columns: the 30 named feature columns plus `target`.
