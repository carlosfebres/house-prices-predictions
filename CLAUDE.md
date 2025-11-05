# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Kaggle House Prices: Advanced Regression Techniques** competition project using R for data science modeling. The goal is to predict residential home sale prices in Ames, Iowa using 80+ explanatory features.

**Dataset Files:**
- `train.csv` - Training data (~1460 houses with SalePrice)
- `test.csv` - Test data (~1459 houses, predict SalePrice)
- `data_description.txt` - Comprehensive documentation of all 80+ features

**Important:** The R model scripts in this repository are currently **under audit and review**. Do not assume they are correct or finalized. Approach them critically and validate all logic, assumptions, and implementations.

## Development Commands

Run individual model scripts:
```bash
Rscript "Model 1 Linear Regression.R"
Rscript modelo2.R
Rscript modelo3.R
```

Each script produces a Kaggle submission CSV file with columns: `Id`, `SalePrice`

## Current Codebase State (Under Review)

### Three Model Approaches
1. **Model 1** - Linear Regression baseline
2. **Model 2** - Decision Tree (CART) with cross-validation and pruning
3. **Model 3** - Random Forest ensemble (1000 trees)

### Common Preprocessing Pattern (Observed Across Scripts)
The scripts appear to share similar data preprocessing steps:
- Missing value imputation (various strategies by feature type)
- Outlier treatment using IQR-based winsorization
- Factor level harmonization between train/test sets
- Log transformation of the target variable (SalePrice)

**Note:** These preprocessing steps need validation during the audit phase.

### Feature Engineering (Advanced Models)
Observed engineered features include:
- Combined square footage features (TotalSF)
- Age-based features (Age, RemodAge)
- Bathroom equivalents (BathEq)
- Interaction terms (e.g., OverallQual × GrLivArea)
- Neighborhood-level aggregations

**Note:** Verify these features are correctly implemented and theoretically sound.

## Data Structure

The dataset contains 80+ features across multiple categories:
- **Physical characteristics**: Lot size, building area, rooms, bathrooms
- **Quality ratings**: Overall quality, condition, material quality
- **Age/temporal**: Year built, year remodeled, year sold
- **Location**: Neighborhood, zoning, proximity features
- **Categorical details**: Building type, foundation, roof, exterior, etc.

See `data_description.txt` for detailed feature definitions and valid values.

**High-missingness features** (many NA values by design):
- PoolQC, MiscFeature, Alley, Fence, FireplaceQu
- Garage-related features (GarageType, GarageFinish, etc.)
- Basement-related features (BsmtQual, BsmtCond, etc.)

## Important Notes for Auditing

When reviewing or modifying the model scripts:

1. **Validate preprocessing logic**: Check that missing value imputation makes sense contextually (e.g., is "None" appropriate for categorical NAs? Should numeric NAs use median or another strategy?)

2. **Review outlier treatment**: Verify that outlier removal/winsorization is applied to appropriate features and doesn't remove valid data points

3. **Check train/test consistency**: Ensure all preprocessing steps applied to training data are correctly applied to test data (factor levels, transformations, feature engineering)

4. **Verify transformations**: Confirm log transformations are properly applied and back-transformed (expm1) for final predictions

5. **Model assumptions**: Validate that each modeling approach's assumptions are met (e.g., linear regression assumptions, tree parameters)

6. **Feature engineering rationale**: Ensure engineered features have clear rationale and are correctly calculated

7. **Data leakage**: Watch for any information from test set leaking into training (especially in neighborhood aggregations or imputation strategies)

## Workflow

The typical workflow appears to be:
1. Load train.csv and test.csv
2. Apply preprocessing pipeline
3. Engineer features (if applicable)
4. Train model on training data
5. Make predictions on test data
6. Back-transform predictions and create submission CSV

Each script is self-contained and can be run independently.
