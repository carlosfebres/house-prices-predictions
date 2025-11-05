# Audit Report: Model 2 Decision Tree.R

## Executive Summary

**Model Approach:** CART Decision Tree with 10-fold Cross-Validation and 1-SE Pruning
**Overall Severity:** 🔴 **CRITICAL** - Contains same data leakage as Model 1
**Issues Found:** 5 Critical, 4 Moderate, 2 Minor

### Quick Verdict
This model shares the **identical preprocessing pipeline** with Model 1 (lines 1-148), inheriting all critical data leakage issues. The decision tree modeling approach (lines 150-185) is well-implemented with proper cross-validation and pruning, but the foundation is compromised by data leakage in preprocessing.

**Recommendation:** Fix critical preprocessing issues before using for Kaggle submissions. The tree modeling itself is sound.

---

## Critical Issues 🔴

### Issue #1: Target Leakage via Neighborhood Median Price (Lines 136-141)

**Severity:** 🔴 CRITICAL
**Location:** Lines 136-141

**Problem:**
```r
nbhd_med <- aggregate(SalePrice_log ~ Neighborhood, data = train, FUN = median)
names(nbhd_med)[2] <- "NbhdMedLogPrice"
train <- merge(train, nbhd_med, by = "Neighborhood", all.x = TRUE, sort = FALSE)
test  <- merge(test,  nbhd_med, by = "Neighborhood", all.x = TRUE, sort = FALSE)
```

**Identical to Model 1 Issue #1.** This creates a feature derived from the target variable (median sale price by neighborhood).

**Why This Matters for Decision Trees:**
- Decision trees can easily split on this feature to achieve artificially low training error
- The tree learns "expensive neighborhoods = expensive houses" (circular reasoning)
- Creates a shortcut that bypasses learning legitimate predictive patterns
- Will overfit to training neighborhoods and fail on new areas

**Impact on Tree Performance:**
- `NbhdMedLogPrice` will likely appear at the top of the tree (high feature importance)
- Tree depth and complexity may be reduced because this feature does most of the work
- Cross-validation error will be optimistic (underestimated)

**Corrected Approach:**

```r
# REMOVE this feature entirely
# Do NOT create NbhdMedLogPrice

# Alternative: Use legitimate neighborhood characteristics
# Example: Number of houses in neighborhood (not derived from target)
train$NbhdHouseCount <- ave(train$Id, train$Neighborhood, FUN = length)
test$NbhdHouseCount <- ave(test$Id, test$Neighborhood, FUN = length)
```

---

### Issue #2: Test Set Numeric Imputation Uses Test Statistics (Line 28)

**Severity:** 🔴 CRITICAL
**Location:** Line 28

**Problem:**
```r
if (col %in% names(test))  test[[col]][is.na(test[[col]])]  <- median(test[[col]], na.rm = TRUE)
```

**Identical to Model 1 Issue #2.** Test set medians leak information from the test set into preprocessing.

**Corrected Code:**

```r
# 2B. Median imputation for numeric variables
num_median <- c("LotFrontage", "GarageYrBlt", "MasVnrArea")

# Calculate medians from TRAINING set only
train_medians <- list()
for (col in num_median) {
  if (col %in% names(train)) {
    train_medians[[col]] <- median(train[[col]], na.rm = TRUE)
    train[[col]][is.na(train[[col]])] <- train_medians[[col]]
  }
}

# Apply TRAINING medians to test set
for (col in num_median) {
  if (col %in% names(test) && !is.null(train_medians[[col]])) {
    test[[col]][is.na(test[[col]])] <- train_medians[[col]]
  }
}
```

---

### Issue #3: Neighborhood-Level LotFrontage Imputation Data Leakage (Lines 31-42)

**Severity:** 🔴 CRITICAL
**Location:** Lines 31-42

**Problem:**
```r
for (df_name in c("train", "test")) {
  df <- get(df_name)
  # ... calculates neighborhood medians separately for each dataset
}
```

**Identical to Model 1 Issue #3.** Neighborhood-level imputation calculated independently for train and test.

**Corrected Code:**

```r
# 2C. Contextual imputation for LotFrontage (using TRAINING neighborhoods only)
if ("LotFrontage" %in% names(train) && "Neighborhood" %in% names(train)) {
  # Calculate from training data
  nbhd_lf_median <- aggregate(LotFrontage ~ Neighborhood,
                               data = train,
                               FUN = function(x) median(x, na.rm = TRUE))
  names(nbhd_lf_median)[2] <- "NbhdLotFrontageMedian"

  # Apply to train
  train <- merge(train, nbhd_lf_median, by = "Neighborhood", all.x = TRUE, sort = FALSE)
  train$LotFrontage <- ifelse(is.na(train$LotFrontage),
                               train$NbhdLotFrontageMedian,
                               train$LotFrontage)
  train$NbhdLotFrontageMedian <- NULL

  # Apply to test
  test <- merge(test, nbhd_lf_median, by = "Neighborhood", all.x = TRUE, sort = FALSE)
  global_lf_median <- median(train$LotFrontage, na.rm = TRUE)
  test$NbhdLotFrontageMedian[is.na(test$NbhdLotFrontageMedian)] <- global_lf_median
  test$LotFrontage <- ifelse(is.na(test$LotFrontage),
                              test$NbhdLotFrontageMedian,
                              test$LotFrontage)
  test$NbhdLotFrontageMedian <- NULL
}
```

---

### Issue #4: Test Set Basement/Garage Imputation Uses Test Statistics (Line 48)

**Severity:** 🔴 CRITICAL
**Location:** Lines 44-49

**Problem:**
```r
for (col in num_extra_test) {
  if (col %in% names(test)) test[[col]][is.na(test[[col]])] <- median(test[[col]], na.rm = TRUE)
}
```

**Identical to Model 1 Issue #4.**

**Corrected Code:**

```r
# 2D. Handle missing basement/garage values (using TRAINING medians)
num_extra_test <- c("BsmtFullBath","BsmtHalfBath","BsmtFinSF1","BsmtFinSF2",
                    "BsmtUnfSF","TotalBsmtSF","GarageCars","GarageArea")

# Calculate from training set
train_medians_extra <- list()
for (col in num_extra_test) {
  if (col %in% names(train)) {
    train_medians_extra[[col]] <- median(train[[col]], na.rm = TRUE)
  }
}

# Apply to test set
for (col in num_extra_test) {
  if (col %in% names(test) && !is.null(train_medians_extra[[col]])) {
    test[[col]][is.na(test[[col]])] <- train_medians_extra[[col]]
  }
}
```

---

### Issue #5: Test Set Mode Imputation Uses Test Statistics (Lines 68-70)

**Severity:** 🔴 CRITICAL
**Location:** Lines 68-70

**Problem:**
```r
for (col in cat_impute_test) {
  if (col %in% names(test)) test[[col]][is.na(test[[col]])] <- get_mode(test[[col]])
}
```

**Identical to Model 1 Issue #5.**

**Corrected Code:**

```r
# 2E. Mode imputation (using TRAINING modes)
get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  if (length(ux) == 0) return(NA)
  ux[which.max(tabulate(match(x, ux)))]
}

# Calculate modes from training set
cat_impute_train <- c("GarageType","GarageFinish","GarageQual","GarageCond",
                      "BsmtExposure","BsmtFinType1","BsmtFinType2",
                      "BsmtQual","BsmtCond","MasVnrType","Electrical")

train_modes <- list()
for (col in cat_impute_train) {
  if (col %in% names(train)) {
    train_modes[[col]] <- get_mode(train[[col]])
    train[[col]][is.na(train[[col]])] <- train_modes[[col]]
  }
}

# Apply to test set
cat_impute_test <- c("GarageType","GarageFinish","GarageQual","GarageCond",
                     "BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual",
                     "BsmtCond","MasVnrType","MSZoning","Utilities","Functional",
                     "Exterior1st","Exterior2nd","KitchenQual","SaleType")

for (col in cat_impute_test) {
  if (col %in% names(test)) {
    if (!is.null(train_modes[[col]])) {
      test[[col]][is.na(test[[col]])] <- train_modes[[col]]
    } else if (col %in% names(train)) {
      mode_val <- get_mode(train[[col]])
      test[[col]][is.na(test[[col]])] <- mode_val
    } else {
      warning(paste("Column", col, "not in training. Using test mode."))
      test[[col]][is.na(test[[col]])] <- get_mode(test[[col]])
    }
  }
}
```

---

## Moderate Issues 🟡

### Issue #6: Aggressive Outlier Treatment (Lines 72-102)

**Severity:** 🟡 MODERATE
**Location:** Lines 72-102

**Problem:**
Outliers replaced with median rather than capped at boundaries.

**Why This Matters for Decision Trees:**
Decision trees are generally **robust to outliers** because they use splits based on thresholds, not distances. However:
- Replacing with median still distorts the data distribution
- Legitimate extreme values (luxury homes) are artificially compressed
- May reduce tree's ability to capture edge cases

**Decision Tree Advantage:**
Unlike linear regression, trees don't assume linear relationships or normality, so outlier treatment is **less critical** for model performance. However, it's still better practice to cap rather than replace.

**Better Approach:**

```r
# STEP 3: Outlier treatment (IMPROVED for trees)
cols_to_fix <- c("OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF",
                 "X1stFlrSF","TotRmsAbvGrd","MasVnrArea","Fireplaces",
                 # Removed: "YearBuilt", "GarageYrBlt" (temporal variables)
                 "BsmtFinSF1","LotFrontage","WoodDeckSF",
                 "X2ndFlrSF","OpenPorchSF","HalfBath")

for (col in cols_to_fix) {
  if (!col %in% names(train)) next

  x_train <- train[[col]]
  Q1 <- quantile(x_train, 0.25, na.rm = TRUE)
  Q3 <- quantile(x_train, 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  if (is.na(IQRv) || IQRv == 0) next

  lo <- Q1 - 1.5 * IQRv
  hi <- Q3 + 1.5 * IQRv

  # CAP at boundaries (better for trees too)
  x_train[!is.na(x_train) & x_train < lo] <- lo
  x_train[!is.na(x_train) & x_train > hi] <- hi
  train[[col]] <- x_train

  if (col %in% names(test)) {
    x_test <- test[[col]]
    x_test[!is.na(x_test) & x_test < lo] <- lo
    x_test[!is.na(x_test) & x_test > hi] <- hi
    test[[col]] <- x_test
  }
}
```

**Alternative for Trees:**
Consider skipping outlier treatment entirely, since trees are naturally robust:
```r
# Decision trees can handle outliers - consider removing this step
# Or only apply to truly erroneous values (e.g., LotFrontage = 0)
```

---

### Issue #7: Treating Temporal Variables as Outliers (Lines 75-76)

**Severity:** 🟡 MODERATE
**Location:** Lines 75-76

**Problem:**
`YearBuilt` and `GarageYrBlt` treated as having outliers.

**Identical to Model 1 Issue #7.** Historic homes are legitimate data points.

**Recommendation:**
```r
# Remove temporal variables from outlier treatment
cols_to_fix <- c("OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF",
                 "X1stFlrSF","TotRmsAbvGrd","MasVnrArea","Fireplaces",
                 # "YearBuilt", "GarageYrBlt",  # REMOVED
                 "BsmtFinSF1","LotFrontage","WoodDeckSF",
                 "X2ndFlrSF","OpenPorchSF","HalfBath")
```

---

### Issue #8: No Missing Prediction Check (Lines 178-180)

**Severity:** 🟡 MODERATE
**Location:** Lines 178-180

**Problem:**
```r
pred_log <- predict(tree, newdata = test)
pred     <- expm1(pred_log)
```

Unlike Model 1, this script doesn't check for NA predictions. While decision trees handle factor levels better than linear models, it's still good practice to verify.

**Why This Matters:**
- If factor harmonization (lines 109-115) fails, predictions could be NA
- Silent failures are harder to debug
- No diagnostic information about prediction quality

**Better Approach:**

```r
# 4) Predict on test set
pred_log <- predict(tree, newdata = test)

# Check for missing predictions
if (any(is.na(pred_log))) {
  n_missing <- sum(is.na(pred_log))
  missing_ids <- test$Id[is.na(pred_log)]
  warning(paste("WARNING:", n_missing, "predictions are NA."))
  warning(paste("IDs:", paste(head(missing_ids, 10), collapse = ", ")))

  # Fallback: use mean
  pred_log[is.na(pred_log)] <- mean(pred_log, na.rm = TRUE)
}

# Back-transform
pred <- expm1(pred_log)

# Sanity checks
cat(sprintf("Prediction range: $%.2f - $%.2f\n", min(pred), max(pred)))
cat(sprintf("Mean prediction: $%.2f\n", mean(pred)))
```

---

### Issue #9: No Model Diagnostics or Performance Metrics (Lines 150-185)

**Severity:** 🟡 MODERATE
**Location:** Model section (lines 150-185)

**Problem:**
The model is trained and predictions made, but there are no:
- Training set RMSE to assess fit quality
- Cross-validation results visualization
- Tree visualization or feature importance
- Comparison of pruned vs. unpruned tree performance

**Why This Matters:**
- Can't assess if pruning improved generalization
- No visibility into which features the tree considers important
- No way to detect overfitting or underfitting
- Can't compare model versions

**Recommended Additions:**

```r
# MODEL 2 — DECISION TREE (with diagnostics)

# 1) Package
library(rpart)
library(rpart.plot)  # For visualization

# 2) Fit tree
set.seed(88)
tree_raw <- rpart(
  formula = form,
  data    = train,
  method  = "anova",
  control = rpart.control(
    minsplit = 10,
    minbucket = 5,
    cp = 0.0005,
    maxdepth = 16,
    xval = 10
  )
)

cat("\n=== Unpruned Tree Summary ===\n")
print(tree_raw)

# 3) Analyze cross-validation results
cat("\n=== Cross-Validation Table ===\n")
printcp(tree_raw)

# Plot CV error vs. complexity
par(mfrow = c(1, 2))
plotcp(tree_raw)

# 4) Prune via 1-SE rule
cp_tab   <- tree_raw$cptable
i_min    <- which.min(cp_tab[,"xerror"])
xerr_min <- cp_tab[i_min,"xerror"]
xstd     <- cp_tab[i_min,"xstd"]
cp_1se   <- cp_tab[which(cp_tab[,"xerror"] <= xerr_min + xstd)[1], "CP"]

cat(sprintf("\nOptimal CP (1-SE rule): %.6f\n", cp_1se))

tree <- prune(tree_raw, cp = cp_1se)

cat("\n=== Pruned Tree Summary ===\n")
print(tree)

# 5) Visualize tree
rpart.plot(tree, main = "Pruned Decision Tree",
           box.palette = "RdYlGn", shadow.col = "gray", nn = TRUE)
par(mfrow = c(1, 1))

# 6) Feature importance
cat("\n=== Variable Importance ===\n")
print(tree$variable.importance)

# 7) Training set performance
train_pred_log <- predict(tree, newdata = train)
train_pred <- expm1(train_pred_log)
train_rmse <- sqrt(mean((train$SalePrice - train_pred)^2))
train_mae <- mean(abs(train$SalePrice - train_pred))

cat("\n=== Training Set Performance ===\n")
cat(sprintf("RMSE: $%.2f\n", train_rmse))
cat(sprintf("MAE:  $%.2f\n", train_mae))

# 8) Predict on test set
pred_log <- predict(tree, newdata = test)

# Check for missing predictions
if (any(is.na(pred_log))) {
  warning(paste("WARNING:", sum(is.na(pred_log)), "predictions are NA."))
  pred_log[is.na(pred_log)] <- mean(pred_log, na.rm = TRUE)
}

pred <- expm1(pred_log)

cat("\n=== Test Set Predictions ===\n")
cat(sprintf("Range: $%.2f - $%.2f\n", min(pred), max(pred)))
cat(sprintf("Mean: $%.2f\n", mean(pred)))
cat(sprintf("Median: $%.2f\n", median(pred)))

# 9) Save submission
submission <- data.frame(Id = test$Id, SalePrice = pred)
write.csv(submission, "Model2_DecisionTree.csv", row.names = FALSE)
cat("\nSaved: Model2_DecisionTree.csv\n")
```

---

## Minor Issues 🔵

### Issue #10: Hyperparameters Not Tuned (Lines 161-167)

**Severity:** 🔵 MINOR
**Location:** Lines 161-167

**Observation:**
```r
control = rpart.control(
  minsplit = 10,
  minbucket = 5,
  cp = 0.0005,
  maxdepth = 16,
  xval = 10
)
```

These hyperparameters are manually set without tuning. While reasonable, they may not be optimal.

**Why This Might Matter:**
- `minsplit = 10` and `minbucket = 5` control tree complexity
- `maxdepth = 16` is quite deep - may allow overfitting
- `cp = 0.0005` is very small - grows a large tree before pruning
- Different values might improve performance

**Tuning Approach:**

```r
# Grid search for optimal hyperparameters
library(caret)

set.seed(88)
tune_grid <- expand.grid(
  cp = c(0.0001, 0.0005, 0.001, 0.005, 0.01)
)

cv_control <- trainControl(method = "cv", number = 10)

tree_tuned <- train(
  form,
  data = train,
  method = "rpart",
  trControl = cv_control,
  tuneGrid = tune_grid,
  control = rpart.control(minsplit = 10, minbucket = 5, maxdepth = 16)
)

cat("\n=== Hyperparameter Tuning Results ===\n")
print(tree_tuned)
plot(tree_tuned)

# Use best model
tree <- tree_tuned$finalModel
```

**Note:** The current approach with 1-SE pruning is actually quite sophisticated and may perform well without explicit tuning.

---

### Issue #11: No Comparison with Baseline (Missing Metric)

**Severity:** 🔵 MINOR
**Location:** Entire script

**Observation:**
No comparison with a simple baseline (e.g., mean price, median price by neighborhood).

**Why This Matters:**
- Hard to know if the model is actually learning useful patterns
- Baseline provides context for model performance
- Useful for detecting bugs (e.g., if model performs worse than mean)

**Recommendation:**

```r
# Add baseline comparison
baseline_pred <- rep(median(train$SalePrice), nrow(test))
cat("\n=== Baseline (Median Price) ===\n")
cat(sprintf("Baseline prediction: $%.2f for all houses\n", median(train$SalePrice)))

# On training set, compare to baseline
baseline_train_rmse <- sqrt(mean((train$SalePrice - median(train$SalePrice))^2))
cat(sprintf("Baseline RMSE (train): $%.2f\n", baseline_train_rmse))
cat(sprintf("Model RMSE (train): $%.2f\n", train_rmse))
cat(sprintf("Improvement: %.1f%%\n",
            100 * (baseline_train_rmse - train_rmse) / baseline_train_rmse))
```

---

## Decision Tree Specific Considerations

### Strengths of This Implementation:

1. **10-Fold Cross-Validation:** Line 166 uses `xval = 10`, which is excellent for assessing generalization
2. **1-SE Pruning Rule:** Lines 171-176 implement the standard pruning approach to prevent overfitting
3. **Reasonable Complexity Parameters:** `minsplit`, `minbucket`, and `cp` are sensible values
4. **Handles Categorical Variables:** Trees naturally handle factors without dummy encoding

### Advantages Over Model 1 (Linear Regression):

- **Non-linear Relationships:** Can capture non-linear effects without explicit transformations
- **Automatic Interactions:** Trees naturally model interactions between variables
- **Robust to Outliers:** Less sensitive to extreme values than linear regression
- **Handles Missing Values:** rpart can handle NA values natively (though this script imputes them)
- **Interpretable:** Tree structure can be visualized and explained

### Disadvantages:

- **High Variance:** Trees can overfit to training data despite pruning
- **Less Smooth Predictions:** Predictions are step functions (constant within leaf nodes)
- **Instability:** Small changes in data can lead to very different trees
- **May Not Capture Additive Effects Well:** If relationship is mostly linear, regression may perform better

---

## Summary of Recommendations

### Must Fix (Critical):
1. ✅ **Remove `NbhdMedLogPrice` feature** - target leakage (Issue #1)
2. ✅ **Fix all imputation to use training statistics only** (Issues #2, #3, #4, #5)

### Should Fix (Moderate):
3. ✅ **Change outlier treatment to boundary capping** - or remove entirely (Issue #6)
4. ✅ **Remove temporal variables from outlier treatment** (Issue #7)
5. ✅ **Add missing prediction checks** (Issue #8)
6. ✅ **Add model diagnostics, visualization, and performance metrics** (Issue #9)

### Consider Fixing (Minor):
7. ⚠️ **Tune hyperparameters with grid search** (Issue #10)
8. ⚠️ **Add baseline comparison** (Issue #11)

---

## Comparison with Model 1

| Aspect | Model 1 (Linear Regression) | Model 2 (Decision Tree) |
|--------|----------------------------|-------------------------|
| **Preprocessing** | Identical (lines 1-148) | Identical (lines 1-148) |
| **Data Leakage** | Yes (5 critical issues) | Yes (5 critical issues) |
| **Cross-Validation** | No | Yes (10-fold) |
| **Pruning/Regularization** | No | Yes (1-SE rule) |
| **Missing Prediction Check** | Yes (lines 162-164) | No |
| **Assumptions** | Many (linearity, normality, etc.) | Few (just need splits) |
| **Interpretability** | Coefficients | Tree structure |
| **Robustness to Outliers** | Low | High |
| **Performance on Non-linear Data** | Poor | Good |

**Verdict:** Model 2's modeling approach is **superior** to Model 1, but both are compromised by the same preprocessing issues.

---

## Expected Impact of Fixes

### After Fixing Critical Issues:
- **Kaggle Score:** Likely to **decrease initially** (removes inflated performance from leakage)
- **Generalization:** Will **improve** - model will learn legitimate patterns
- **Tree Complexity:** May **increase** - without the NbhdMedLogPrice shortcut, tree needs more splits
- **Feature Importance:** Will **shift** - other features will become more important

### After Adding Diagnostics:
- Better understanding of model behavior
- Easier to debug and improve
- Can identify which features matter most
- Can compare pruned vs. unpruned performance

---

## Additional Resources

- [CART Algorithm Explained](https://online.stat.psu.edu/stat508/lesson/9/9.1)
- [rpart Package Documentation](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf)
- [Pruning Decision Trees](https://www.statlearning.com/) - Chapter 8
- [Feature Engineering for Trees](https://www.kaggle.com/learn/feature-engineering)

---

**Audit Completed:** 2025-11-05
**Auditor:** Claude (AI Assistant)
**Next Steps:** Fix critical preprocessing issues, then compare Model 2 performance with fixed Model 1
