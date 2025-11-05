# Audit Report: Model 1 Linear Regression.R

## Executive Summary

**Model Approach:** Multiple Linear Regression with log-transformed target
**Overall Severity:** 🔴 **CRITICAL** - Contains data leakage that invalidates results
**Issues Found:** 5 Critical, 4 Moderate, 3 Minor

### Quick Verdict
This model contains **systematic data leakage** that will produce inflated performance metrics and unreliable Kaggle predictions. The most serious issue is the `NbhdMedLogPrice` feature (lines 136-141), which leaks target information into the model. Additionally, test set preprocessing uses test set statistics instead of training statistics, further compromising validity.

**Recommendation:** Do not use this model for competition submissions until critical issues are resolved.

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

This creates a feature that encodes the median log sale price by neighborhood and uses it as a predictor. **This is target leakage** - you're using aggregated information derived from the target variable as a feature.

**Why This Matters:**
- Creates unrealistically high R² and low RMSE on training data
- The model learns "houses in expensive neighborhoods are expensive" (circular reasoning)
- Overfits to training neighborhoods
- Poor generalization to new data or neighborhoods not seen in training
- Violates fundamental ML principle: features should not contain target information

**Corrected Approach:**

```r
# OPTION A: Remove the feature entirely (safest)
# Simply don't create NbhdMedLogPrice

# OPTION B: Use legitimate neighborhood features instead
# Example: Create neighborhood characteristics that don't use target
train$NbhdHouseCount <- ave(train$Id, train$Neighborhood, FUN = length)
test$NbhdHouseCount <- ave(test$Id, test$Neighborhood, FUN = length)

# Or use external data (e.g., census data, crime rates, school ratings)
# that don't leak information from the target variable
```

---

### Issue #2: Test Set Numeric Imputation Uses Test Statistics (Line 28)

**Severity:** 🔴 CRITICAL
**Location:** Line 28

**Problem:**
```r
if (col %in% names(test))  test[[col]][is.na(test[[col]])]  <- median(test[[col]], na.rm = TRUE)
```

The test set missing values are imputed using the median **of the test set itself**, not the training set. This is data leakage.

**Why This Matters:**
- In real-world deployment, you won't have access to test set statistics
- Violates train/test separation - test data influences preprocessing
- Can lead to distribution shifts between development and production
- Overestimates model performance

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
  if ("LotFrontage" %in% names(df) && "Neighborhood" %in% names(df)) {
    df$LotFrontage <- ifelse(
      is.na(df$LotFrontage),
      ave(df$LotFrontage, df$Neighborhood, FUN = function(x) median(x, na.rm = TRUE)),
      df$LotFrontage
    )
  }
  assign(df_name, df)
}
```

This code calculates neighborhood-level medians **separately for train and test**, meaning test set imputation uses test set statistics.

**Corrected Code:**

```r
# 2C. Contextual imputation for LotFrontage (Neighborhood-level median)
# Calculate neighborhood medians from TRAINING set
if ("LotFrontage" %in% names(train) && "Neighborhood" %in% names(train)) {
  nbhd_lf_median <- aggregate(LotFrontage ~ Neighborhood,
                               data = train,
                               FUN = function(x) median(x, na.rm = TRUE))
  names(nbhd_lf_median)[2] <- "NbhdLotFrontageMedian"

  # Merge with train
  train <- merge(train, nbhd_lf_median, by = "Neighborhood", all.x = TRUE, sort = FALSE)
  train$LotFrontage <- ifelse(is.na(train$LotFrontage),
                               train$NbhdLotFrontageMedian,
                               train$LotFrontage)
  train$NbhdLotFrontageMedian <- NULL  # Remove helper column

  # Merge with test using TRAINING neighborhoods
  test <- merge(test, nbhd_lf_median, by = "Neighborhood", all.x = TRUE, sort = FALSE)

  # For neighborhoods not in training, use overall training median
  global_lf_median <- median(train$LotFrontage, na.rm = TRUE)
  test$NbhdLotFrontageMedian[is.na(test$NbhdLotFrontageMedian)] <- global_lf_median

  test$LotFrontage <- ifelse(is.na(test$LotFrontage),
                              test$NbhdLotFrontageMedian,
                              test$LotFrontage)
  test$NbhdLotFrontageMedian <- NULL  # Remove helper column
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

Same issue as #2 - using test set statistics for imputation.

**Corrected Code:**

```r
# 2D. Handle missing basement/garage values in test set (numeric)
num_extra_test <- c("BsmtFullBath","BsmtHalfBath","BsmtFinSF1","BsmtFinSF2",
                    "BsmtUnfSF","TotalBsmtSF","GarageCars","GarageArea")

# Calculate medians from TRAINING set
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

Categorical mode is calculated from test set, not training set.

**Corrected Code:**

```r
# 2E. Mode imputation for categorical variables
get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  if (length(ux) == 0) return(NA)
  ux[which.max(tabulate(match(x, ux)))]
}

# Calculate modes from TRAINING set
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

# Apply to test set (extended list including test-only missing columns)
cat_impute_test <- c("GarageType","GarageFinish","GarageQual","GarageCond",
                     "BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual",
                     "BsmtCond","MasVnrType","MSZoning","Utilities","Functional",
                     "Exterior1st","Exterior2nd","KitchenQual","SaleType")

for (col in cat_impute_test) {
  if (col %in% names(test)) {
    # Use training mode if available, otherwise calculate from test (with warning)
    if (!is.null(train_modes[[col]])) {
      test[[col]][is.na(test[[col]])] <- train_modes[[col]]
    } else if (col %in% names(train)) {
      # Column exists in train but wasn't in impute list - calculate now
      mode_val <- get_mode(train[[col]])
      test[[col]][is.na(test[[col]])] <- mode_val
    } else {
      # Column only in test - last resort, use test mode with warning
      warning(paste("Column", col, "not in training set. Using test set mode."))
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
Outliers are **replaced with the median** rather than capped at boundaries:
```r
out_train <- !is.na(x_train) & (x_train < lo | x_train > hi)
if (any(out_train)) x_train[out_train] <- med
```

**Why This Matters:**
- Destroys legitimate extreme values (e.g., luxury homes with large living areas)
- All outliers collapse to the same median value, reducing variance artificially
- Can introduce bias by pulling extreme values toward center
- Winsorization (capping at boundaries) is generally preferred

**Better Approach - Winsorization (Capping):**

```r
# STEP 3: Outlier treatment (IMPROVED: Cap at boundaries instead of median)
cols_to_fix <- c("OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF",
                 "X1stFlrSF","TotRmsAbvGrd","MasVnrArea","Fireplaces",
                 "BsmtFinSF1","LotFrontage","WoodDeckSF",
                 "X2ndFlrSF","OpenPorchSF","HalfBath")
# Note: Removed YearBuilt and GarageYrBlt - see Issue #7

for (col in cols_to_fix) {
  if (!col %in% names(train)) next

  x_train <- train[[col]]
  Q1 <- quantile(x_train, 0.25, na.rm = TRUE)
  Q3 <- quantile(x_train, 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  if (is.na(IQRv) || IQRv == 0) next

  lo <- Q1 - 1.5 * IQRv
  hi <- Q3 + 1.5 * IQRv

  # CAP at boundaries instead of replacing with median
  x_train[!is.na(x_train) & x_train < lo] <- lo
  x_train[!is.na(x_train) & x_train > hi] <- hi
  train[[col]] <- x_train

  # Apply same bounds to test set
  if (col %in% names(test)) {
    x_test <- test[[col]]
    x_test[!is.na(x_test) & x_test < lo] <- lo
    x_test[!is.na(x_test) & x_test > hi] <- hi
    test[[col]] <- x_test
  }
}
```

**Alternative Approach - Log Transformation:**
For highly skewed variables like `GrLivArea`, consider log transformation instead:
```r
train$GrLivArea_log <- log1p(train$GrLivArea)
test$GrLivArea_log <- log1p(test$GrLivArea)
```

---

### Issue #7: Treating Temporal Variables as Outliers (Lines 75-76)

**Severity:** 🟡 MODERATE
**Location:** Lines 75-76 in `cols_to_fix`

**Problem:**
`YearBuilt` and `GarageYrBlt` are included in outlier treatment. These are temporal variables where old houses are **legitimately old**, not anomalies.

**Why This Matters:**
- Historic homes (e.g., built in 1880) are valid data points, not errors
- Replacing old years with median artificially ages/de-ages houses
- You already create `Age` and `RemodAge` features (lines 123-127), which are better representations

**Recommendation:**
```r
# Remove YearBuilt and GarageYrBlt from outlier treatment
cols_to_fix <- c("OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF",
                 "X1stFlrSF","TotRmsAbvGrd","MasVnrArea","Fireplaces",
                 # "YearBuilt", "GarageYrBlt",  # REMOVED - temporal variables
                 "BsmtFinSF1","LotFrontage","WoodDeckSF",
                 "X2ndFlrSF","OpenPorchSF","HalfBath")

# The Age and RemodAge features (lines 123-127) already capture temporal information appropriately
```

---

### Issue #8: Missing Prediction Handling Without Root Cause Analysis (Lines 162-164)

**Severity:** 🟡 MODERATE
**Location:** Lines 162-164

**Problem:**
```r
if (any(is.na(pred_log))) {
  pred_log[is.na(pred_log)] <- mean(pred_log, na.rm = TRUE)
}
```

Missing predictions are silently filled with the mean without investigating **why** they're missing.

**Why This Matters:**
- Missing predictions usually indicate factor level mismatches (test has levels not in train)
- Silent imputation masks underlying data quality issues
- May predict wrong prices for specific houses

**Better Approach:**

```r
# 4) Predict on the test set
pred_log <- predict(lm_model, newdata = test)

# Check for and diagnose missing predictions
if (any(is.na(pred_log))) {
  n_missing <- sum(is.na(pred_log))
  missing_ids <- test$Id[is.na(pred_log)]

  warning(paste("WARNING:", n_missing, "predictions are NA."))
  warning(paste("Missing prediction IDs:", paste(head(missing_ids, 10), collapse = ", ")))

  # Diagnose factor level issues
  for (col in names(test)) {
    if (is.factor(test[[col]])) {
      test_levels <- levels(test[[col]])
      train_levels <- levels(train[[col]])
      new_levels <- setdiff(test_levels, train_levels)
      if (length(new_levels) > 0) {
        warning(paste("Column", col, "has new levels in test:",
                     paste(new_levels, collapse = ", ")))
      }
    }
  }

  # Fallback: use mean, but log it
  pred_log[is.na(pred_log)] <- mean(pred_log, na.rm = TRUE)
  cat("Filled", n_missing, "missing predictions with mean.\n")
}
```

---

### Issue #9: No Model Diagnostics or Validation (Lines 150-173)

**Severity:** 🟡 MODERATE
**Location:** Model section (lines 150-173)

**Problem:**
The model is trained and predictions are made, but there are no:
- Residual plots to check assumptions
- Cross-validation to assess generalization
- Training set performance metrics (RMSE, R²)
- Multicollinearity checks (VIF)

**Why This Matters:**
- Linear regression has assumptions (linearity, normality, homoscedasticity)
- Can't assess if the model is performing well or poorly
- No way to detect overfitting or underfitting
- Multicollinearity between predictors can inflate variance

**Recommended Additions:**

```r
# MODEL 1 — LINEAR REGRESSION (with diagnostics)

# 1) Fit multiple linear regression
lm_model <- lm(form, data = train)

# 2) Model diagnostics
cat("\n=== Model Diagnostics ===\n")

# Summary statistics
summary(lm_model)

# Residual plots
par(mfrow = c(2, 2))
plot(lm_model)
par(mfrow = c(1, 1))

# Check multicollinearity (requires 'car' package)
if (require(car, quietly = TRUE)) {
  vif_values <- vif(lm_model)
  cat("\nVariance Inflation Factors (VIF > 10 indicates multicollinearity):\n")
  print(vif_values[vif_values > 5])  # Show problematic VIFs
}

# 3) Cross-validation
if (require(caret, quietly = TRUE)) {
  set.seed(123)
  cv_control <- trainControl(method = "cv", number = 10)
  cv_model <- train(form, data = train, method = "lm", trControl = cv_control)
  cat("\n=== Cross-Validation Results ===\n")
  print(cv_model$results)
}

# 4) Training set performance
train_pred_log <- predict(lm_model, newdata = train)
train_pred <- expm1(train_pred_log)
train_rmse <- sqrt(mean((train$SalePrice - train_pred)^2))
train_mae <- mean(abs(train$SalePrice - train_pred))

cat("\n=== Training Set Performance ===\n")
cat(sprintf("RMSE: $%.2f\n", train_rmse))
cat(sprintf("MAE:  $%.2f\n", train_mae))
cat(sprintf("R²:   %.4f\n", summary(lm_model)$r.squared))

# 5) Predict on test set
pred_log <- predict(lm_model, newdata = test)

# Handle potential missing predictions (with diagnostics)
if (any(is.na(pred_log))) {
  n_missing <- sum(is.na(pred_log))
  warning(paste("WARNING:", n_missing, "predictions are NA."))
  pred_log[is.na(pred_log)] <- mean(pred_log, na.rm = TRUE)
}

# 6) Back-transform predictions to original scale
pred <- expm1(pred_log)

# 7) Save submission
submission <- data.frame(Id = test$Id, SalePrice = pred)
write.csv(submission, "Model1_LinearRegression.csv", row.names = FALSE)
cat("\nSaved: Model1_LinearRegression.csv\n")
```

---

## Minor Issues 🔵

### Issue #10: Merge Operation Row Order Risk (Lines 138-139)

**Severity:** 🔵 MINOR
**Location:** Lines 138-139

**Problem:**
```r
train <- merge(train, nbhd_med, by = "Neighborhood", all.x = TRUE, sort = FALSE)
test  <- merge(test,  nbhd_med, by = "Neighborhood", all.x = TRUE, sort = FALSE)
```

While `sort=FALSE` should preserve order, `merge()` can sometimes reorder rows if the merge key appears multiple times.

**Better Approach:**

```r
# Safer approach: Use match() to preserve exact row order
nbhd_med <- aggregate(SalePrice_log ~ Neighborhood, data = train, FUN = median)
names(nbhd_med)[2] <- "NbhdMedLogPrice"

# Use match to preserve order
train$NbhdMedLogPrice <- nbhd_med$NbhdMedLogPrice[match(train$Neighborhood, nbhd_med$Neighborhood)]
test$NbhdMedLogPrice <- nbhd_med$NbhdMedLogPrice[match(test$Neighborhood, nbhd_med$Neighborhood)]

# Handle neighborhoods in test not in train
global_med <- median(train$SalePrice_log, na.rm = TRUE)
test$NbhdMedLogPrice[is.na(test$NbhdMedLogPrice)] <- global_med
```

**Note:** This issue is moot if you remove the `NbhdMedLogPrice` feature (recommended per Issue #1).

---

### Issue #11: No Interaction Term Justification (Line 132)

**Severity:** 🔵 MINOR
**Location:** Lines 132-133

**Observation:**
```r
train$OverallQual_GrLivArea <- train$OverallQual * train$GrLivArea
test$OverallQual_GrLivArea  <- test$OverallQual  * test$GrLivArea
```

This interaction term assumes that the effect of `GrLivArea` on price depends on `OverallQual` (or vice versa).

**Why This Might Be Good:**
- High-quality large homes may command premium prices beyond additive effects
- Interaction terms can capture non-linear relationships

**Recommendation:**
- Justified conceptually, but should be validated with EDA or domain knowledge
- Consider testing model with and without this interaction
- May want to add other interactions (e.g., `Neighborhood × OverallQual`)

---

### Issue #12: Formula Includes Raw `GrLivArea` and Derived `TotalSF` (Line 144-145)

**Severity:** 🔵 MINOR
**Location:** Lines 144-146

**Observation:**
```r
form <- SalePrice_log ~ OverallQual + TotalSF + GrLivArea + ...
```

Where `TotalSF = TotalBsmtSF + X1stFlrSF + X2ndFlrSF` (line 120).

**Potential Issue:**
- `GrLivArea` typically equals `X1stFlrSF + X2ndFlrSF` (above-grade living area)
- Including both `TotalSF` (which contains `X1stFlrSF + X2ndFlrSF`) and `GrLivArea` may introduce multicollinearity
- Check VIF values to confirm

**Recommendation:**
```r
# Check if GrLivArea is redundant
# If multicollinearity is high, consider:
# Option 1: Remove GrLivArea, keep TotalSF
# Option 2: Remove TotalSF, keep GrLivArea + TotalBsmtSF separately
# Option 3: Use only TotalSF (most parsimonious)

form <- SalePrice_log ~ OverallQual + TotalSF + GarageCars +
  Age + RemodAge + LotFrontage + Neighborhood + HouseStyle + GarageType +
  SaleCondition + BathEq + OverallQual_GrLivArea  # Removed GrLivArea to avoid redundancy
```

---

## Summary of Recommendations

### Must Fix (Critical):
1. ✅ **Remove `NbhdMedLogPrice` feature** - target leakage (Issue #1)
2. ✅ **Fix all imputation to use training statistics only** (Issues #2, #3, #4, #5)

### Should Fix (Moderate):
3. ✅ **Change outlier treatment from median replacement to boundary capping** (Issue #6)
4. ✅ **Remove temporal variables from outlier treatment** (Issue #7)
5. ✅ **Add diagnostic logging for missing predictions** (Issue #8)
6. ✅ **Add model diagnostics and cross-validation** (Issue #9)

### Consider Fixing (Minor):
7. ⚠️ **Use safer merging approach** (Issue #10)
8. ⚠️ **Validate interaction terms with EDA** (Issue #11)
9. ⚠️ **Check multicollinearity between TotalSF and GrLivArea** (Issue #12)

---

## Complete Corrected Script Template

Due to length constraints, a complete corrected script would:
1. Calculate all imputation statistics (medians, modes, neighborhood aggregates) from training set
2. Store these in named lists or data structures
3. Apply them consistently to both train and test sets
4. Remove the `NbhdMedLogPrice` feature
5. Use boundary capping for outliers
6. Add comprehensive diagnostics
7. Include cross-validation

**Estimated Impact on Kaggle Score:**
- Fixing data leakage: Likely **worse** short-term score (removes inflated performance)
- Long-term: **Better** generalization and more honest assessment
- Properly prevents overfitting to training neighborhoods

---

## Additional Resources

- [Kaggle: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)
- [Feature Engineering Best Practices](https://www.feat uretools.com/)
- [Linear Regression Assumptions](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/)
- [Handling Missing Data in R](https://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/)

---

**Audit Completed:** 2025-11-05
**Auditor:** Claude (AI Assistant)
**Next Steps:** Review and implement critical fixes before using for Kaggle submission
