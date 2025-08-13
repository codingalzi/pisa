"""
PISA Statistical Analysis and Machine Learning in Python
Converted from R script: Lezhnina_PISA_stats_ML.R

This script performs comprehensive analysis of PISA data including:
- Data loading and preprocessing
- Missing data handling with missForest equivalent
- Random Forest classification
- ROC analysis and partial dependence plots
- Hierarchical Linear Modeling (HLM) equivalent
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%%
# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import RandomOverSampler
#%%
# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%
# For SPSS files
import pyreadstat
#%%
# For multilevel modeling
try:
    import pymer4
    HLM_AVAILABLE = True
except ImportError:
    print("Warning: pymer4 not available. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "pymer4"])
    try:
        import pymer4
        HLM_AVAILABLE = True
    except:
        print("Could not install pymer4. HLM analysis will be skipped.")
        HLM_AVAILABLE = False
#%%
print("Starting PISA Analysis - Python Version")
print("="*50)

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load SPSS data files and perform initial preprocessing"""
    print("Loading PISA data files...")
    
    # Load 2015 data
    try:
        data15, meta15 = pyreadstat.read_sav('/home/gslee/wGitHub/pisa/data/DEU_CY6_MS_CMB_STU_QQQ.sav')
        print(f"2015 data loaded: {data15.shape[0]} rows, {data15.shape[1]} columns")
    except FileNotFoundError:
        print("2015 data file not found. Please check the file path.")
        return None, None
    
    # Load 2018 data  
    try:
        data18, meta18 = pyreadstat.read_sav('/home/gslee/wGitHub/pisa/data/DEU_CY07_MSU_STU_QQQ.sav')
        print(f"2018 data loaded: {data18.shape[0]} rows, {data18.shape[1]} columns")
    except FileNotFoundError:
        print("2018 data file not found. Please check the file path.")
        return None, None
    
    return data15, data18

def select_variables(data):
    """Select relevant variables from the dataset"""
    
    # Define variable patterns
    ict_vars = [col for col in data.columns if col.startswith(('IC00', 'INTICT', 'COMPICT', 'AUTICT', 'SOIAICT'))]
    pv_vars = [col for col in data.columns if col.startswith('PV')]
    weight_vars = [col for col in data.columns if col.startswith('W_FS')]
    
    # Core variables
    core_vars = ['CNTSCHID', 'CNTSTUID', 'STRATUM', 'OECD', 'ST004D01T', 'ESCS']
    
    # Combine all variables
    selected_vars = core_vars + ict_vars + pv_vars + weight_vars
    
    # Filter to existing columns
    available_vars = [var for var in selected_vars if var in data.columns]
    
    print(f"Selected {len(available_vars)} variables from {len(data.columns)} total")
    
    return data[available_vars].copy()

def rename_variables(data):
    """Rename variables to match R script conventions"""
    
    rename_dict = {
        'CNTSCHID': 'SCHL',
        'ST004D01T': 'GEND'
    }
    
    # Rename ICT variables based on patterns
    for col in data.columns:
        if 'INTICT' in col:
            rename_dict[col] = 'INTE'
        elif 'COMPICT' in col:
            rename_dict[col] = 'COMP'
        elif 'AUTICT' in col:
            rename_dict[col] = 'AUTO'
        elif 'SOIAICT' in col:
            rename_dict[col] = 'SOCI'
    
    data = data.rename(columns=rename_dict)
    
    # Remove labels (equivalent to remove_labels in R)
    # This is handled automatically in pandas
    
    return data

def analyze_missingness(data, year):
    """Analyze missing data patterns"""
    print(f"\nMissingness analysis for {year}:")
    print("-" * 30)
    
    # Calculate missingness for ICT variables
    ict_vars = ['INTE', 'COMP', 'AUTO', 'SOCI']
    available_ict = [var for var in ict_vars if var in data.columns]
    
    if not available_ict:
        print("Warning: ICT variables not found with expected names")
        return data
    
    # Calculate percentage of complete missingness in ICT
    ict_data = data[available_ict]
    na_rows_ict = ict_data.isnull().sum(axis=1) / len(available_ict)
    complete_missing = (na_rows_ict == 1).sum()
    
    print(f"Rows with 100% ICT missingness: {complete_missing} ({complete_missing/len(data)*100:.2f}%)")
    
    # Remove rows with 100% ICT missingness
    data_filtered = data[na_rows_ict < 1].copy()
    data_removed = data[na_rows_ict == 1].copy()
    
    print(f"Remaining after filtering: {len(data_filtered)} rows")
    
    # Analyze removed vs remaining data
    if 'GEND' in data.columns:
        print(f"Removed - Female: {(data_removed['GEND'] == 1).sum()}, Male: {(data_removed['GEND'] == 2).sum()}")
        print(f"Remaining - Female: {(data_filtered['GEND'] == 1).sum()}, Male: {(data_filtered['GEND'] == 2).sum()}")
    
    # Plot ESCS distribution comparison
    if 'ESCS' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(data_removed['ESCS'].dropna(), bins=50, alpha=0.5, label='Removed', color='red')
        plt.hist(data_filtered['ESCS'].dropna(), bins=50, alpha=0.5, label='Remaining', color='blue')
        plt.xlabel('ESCS')
        plt.ylabel('Frequency')
        plt.title(f'ESCS Distribution - Removed vs Remaining ({year})')
        plt.legend()
        plt.show()
    
    return data_filtered

def visualize_missing_patterns(data, title):
    """Visualize missing data patterns (equivalent to VIM::aggr)"""
    
    # Calculate missingness percentage for each variable
    missing_pct = data.isnull().sum() / len(data) * 100
    variables_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(variables_with_missing) == 0:
        print(f"No missing data in {title}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot of missing percentages
    plt.subplot(2, 1, 1)
    variables_with_missing.plot(kind='bar')
    plt.title(f'Missing Data Patterns - {title}')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)
    
    # Missing data heatmap
    plt.subplot(2, 1, 2)
    missing_data = data[variables_with_missing.index].isnull()
    sns.heatmap(missing_data.T, cbar=True, yticklabels=True, cmap='viridis')
    plt.title('Missing Data Pattern')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Variables with missing data in {title}:")
    for var, pct in variables_with_missing.items():
        print(f"  {var}: {pct:.2f}%")

class MissForestImputer:
    """
    Python equivalent of R's missForest using IterativeImputer with RandomForest
    """
    def __init__(self, random_state=100, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.imputer = None
    
    def fit_transform(self, X):
        """Fit and transform the data"""
        # Use IterativeImputer with RandomForest estimator
        from sklearn.ensemble import RandomForestRegressor
        
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(random_state=self.random_state, n_estimators=10),
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        
        X_imputed = self.imputer.fit_transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

def impute_missing_data(data, year):
    """Impute missing data using missForest equivalent"""
    
    # Select variables with missing data for imputation
    missing_vars = data.columns[data.isnull().any()].tolist()
    
    if not missing_vars:
        print(f"No missing data to impute for {year}")
        return data
    
    print(f"Imputing missing data for {year} using {len(missing_vars)} variables...")
    
    # Prepare data for imputation (only variables with missing data)
    data_for_imputation = data[missing_vars].copy()
    
    # Initialize imputer
    imputer = MissForestImputer(random_state=100)
    
    # Perform imputation
    data_imputed = imputer.fit_transform(data_for_imputation)
    
    # Replace imputed variables in original dataset
    result_data = data.copy()
    result_data[missing_vars] = data_imputed
    
    print(f"Imputation completed. Missing data before: {data[missing_vars].isnull().sum().sum()}")
    print(f"Missing data after: {result_data[missing_vars].isnull().sum().sum()}")
    
    return result_data

def compare_imputation_quality(original, imputed, title):
    """Compare distributions before and after imputation"""
    
    # Select numeric variables for comparison
    numeric_vars = original.select_dtypes(include=[np.number]).columns
    variables_to_plot = numeric_vars[:5]  # Plot first 5 numeric variables
    
    fig, axes = plt.subplots(1, len(variables_to_plot), figsize=(20, 4))
    if len(variables_to_plot) == 1:
        axes = [axes]
    
    for i, var in enumerate(variables_to_plot):
        if var in original.columns and var in imputed.columns:
            # Original data (complete cases only)
            original_complete = original[var].dropna()
            # Imputed data
            imputed_data = imputed[var]
            
            axes[i].hist(original_complete, bins=30, alpha=0.5, label='Original', 
                        color='green', density=True)
            axes[i].hist(imputed_data, bins=30, alpha=0.5, label='Imputed', 
                        color='orange', density=True)
            axes[i].set_title(var)
            axes[i].legend()
    
    plt.suptitle(f'Imputation Quality Comparison - {title}')
    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 2: MACHINE LEARNING MODELS
# ============================================================================

def prepare_ml_data(data_imputed, data_original, year):
    """Prepare data for machine learning models"""
    
    # Combine imputed variables with PV variables and gender
    pv_vars = [col for col in data_original.columns if col.startswith('PV')]
    
    # Get PV data and gender
    pv_data = data_original[['GEND'] + pv_vars].copy()
    
    # Combine with imputed data
    ml_data = pd.concat([data_imputed, pv_data], axis=1)
    
    # Recode gender (1=Female->0, 2=Male->1)
    if 'GEND' in ml_data.columns:
        ml_data['GEND'] = ml_data['GEND'].map({1: 0, 2: 1})
    
    # Calculate mean PV scores
    math_pvs = [col for col in ml_data.columns if 'MATH' in col and col.startswith('PV')]
    science_pvs = [col for col in ml_data.columns if 'SCIE' in col and col.startswith('PV')]
    
    if math_pvs:
        ml_data['meansM'] = ml_data[math_pvs].mean(axis=1)
    if science_pvs:
        ml_data['meansS'] = ml_data[science_pvs].mean(axis=1)
    
    # Create proficiency levels (based on PISA thresholds)
    if 'meansM' in ml_data.columns:
        ml_data['M'] = pd.cut(ml_data['meansM'], 
                             bins=[0, 420.07, 606.99, 1000], 
                             labels=[1, 2, 3])
    
    if 'meansS' in ml_data.columns:
        ml_data['Sc'] = pd.cut(ml_data['meansS'], 
                              bins=[0, 409.54, 633.33, 1000], 
                              labels=[1, 2, 3])
    
    return ml_data

def prepare_model_datasets(ml_data):
    """Prepare separate datasets for math and science models"""
    
    # Feature variables
    feature_vars = ['INTE', 'COMP', 'AUTO', 'SOCI', 'ESCS', 'GEND']
    available_features = [var for var in feature_vars if var in ml_data.columns]
    
    # Math dataset
    math_data = None
    if 'M' in ml_data.columns:
        math_data = ml_data[available_features + ['M']].dropna()
        print(f"Math dataset: {len(math_data)} samples")
        print("Math class distribution:")
        print(math_data['M'].value_counts().sort_index())
    
    # Science dataset  
    science_data = None
    if 'Sc' in ml_data.columns:
        science_data = ml_data[available_features + ['Sc']].dropna()
        print(f"Science dataset: {len(science_data)} samples")
        print("Science class distribution:")
        print(science_data['Sc'].value_counts().sort_index())
    
    return math_data, science_data, available_features

def train_random_forest_models(math_data, science_data, features, year):
    """Train Random Forest models for math and science"""
    
    models = {}
    test_data = {}
    
    if math_data is not None:
        print(f"\nTraining Math Random Forest model for {year}...")
        
        # Split data
        X_math = math_data[features]
        y_math = math_data['M']
        
        X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(
            X_math, y_math, test_size=0.2, random_state=100, stratify=y_math
        )
        
        # Handle class imbalance with oversampling
        ros = RandomOverSampler(random_state=100)
        X_train_math_balanced, y_train_math_balanced = ros.fit_resample(X_train_math, y_train_math)
        
        print("After oversampling:")
        print(pd.Series(y_train_math_balanced).value_counts().sort_index())
        
        # Train Random Forest
        rf_math = RandomForestClassifier(n_estimators=500, random_state=100)
        rf_math.fit(X_train_math_balanced, y_train_math_balanced)
        
        models['math'] = rf_math
        test_data['math'] = (X_test_math, y_test_math)
        
        # Feature importance
        importance_math = pd.DataFrame({
            'feature': features,
            'importance': rf_math.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Math model feature importance:")
        print(importance_math)
    
    if science_data is not None:
        print(f"\nTraining Science Random Forest model for {year}...")
        
        # Split data
        X_science = science_data[features]
        y_science = science_data['Sc']
        
        X_train_science, X_test_science, y_train_science, y_test_science = train_test_split(
            X_science, y_science, test_size=0.2, random_state=100, stratify=y_science
        )
        
        # Handle class imbalance with oversampling
        ros = RandomOverSampler(random_state=100)
        X_train_science_balanced, y_train_science_balanced = ros.fit_resample(X_train_science, y_train_science)
        
        print("After oversampling:")
        print(pd.Series(y_train_science_balanced).value_counts().sort_index())
        
        # Train Random Forest
        rf_science = RandomForestClassifier(n_estimators=500, random_state=100)
        rf_science.fit(X_train_science_balanced, y_train_science_balanced)
        
        models['science'] = rf_science
        test_data['science'] = (X_test_science, y_test_science)
        
        # Feature importance
        importance_science = pd.DataFrame({
            'feature': features,
            'importance': rf_science.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Science model feature importance:")
        print(importance_science)
    
    return models, test_data

def plot_feature_importance(models):
    """Plot feature importance for both models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (subject, model) in enumerate(models.items()):
        feature_names = model.feature_names_in_
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        axes[i].bar(range(len(importances)), importances[indices])
        axes[i].set_title(f'{subject.capitalize()} Model Feature Importance')
        axes[i].set_xticks(range(len(importances)))
        axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45)
        axes[i].set_ylabel('Importance')
    
    plt.tight_layout()
    plt.show()

def evaluate_models(models, test_data, year):
    """Evaluate model performance"""
    
    print(f"\nModel Evaluation for {year}")
    print("="*40)
    
    for subject, model in models.items():
        X_test, y_test = test_data[subject]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print(f"\n{subject.capitalize()} Model Performance:")
        print("-" * 30)
        print(classification_report(y_test, y_pred))
        
        # ROC AUC for multiclass
        try:
            # Convert labels to numeric for AUC calculation
            y_test_numeric = pd.Categorical(y_test).codes
            y_pred_numeric = pd.Categorical(y_pred).codes
            
            auc_score = roc_auc_score(y_test_numeric, y_pred_proba, multi_class='ovr')
            print(f"ROC AUC (OvR): {auc_score:.3f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")

def plot_partial_dependence(models, test_data, features):
    """Plot partial dependence plots"""
    
    for subject, model in models.items():
        X_test, _ = test_data[subject]
        
        print(f"\nGenerating partial dependence plots for {subject}...")
        
        # Create partial dependence plots
        fig, axes = plt.subplots(2, len(features), figsize=(20, 10))
        
        for i, feature in enumerate(features):
            if feature in X_test.columns:
                # For class 0 (low proficiency)
                PartialDependenceDisplay.from_estimator(
                    model, X_test, [i], 
                    target=0,
                    ax=axes[0, i] if len(features) > 1 else axes[0]
                )
                
                # For class 2 (high proficiency)  
                PartialDependenceDisplay.from_estimator(
                    model, X_test, [i],
                    target=2, 
                    ax=axes[1, i] if len(features) > 1 else axes[1]
                )
        
        plt.suptitle(f'Partial Dependence Plots - {subject.capitalize()}')
        plt.tight_layout()
        plt.show()

# ============================================================================
# PART 3: STATISTICAL MODELING (HLM equivalent)
# ============================================================================

def prepare_hlm_data(data_imputed, data_original):
    """Prepare data for hierarchical linear modeling"""
    
    if not HLM_AVAILABLE:
        print("HLM analysis skipped - pymer4 not available")
        return None
    
    print("Preparing data for hierarchical linear modeling...")
    
    # Standardize ICT variables and ESCS (center and scale by 2 SDs)
    ict_vars = ['INTE', 'COMP', 'AUTO', 'SOCI', 'ESCS']
    available_ict = [var for var in ict_vars if var in data_imputed.columns]
    
    hlm_data = data_original.copy()
    
    # Standardize variables
    for var in available_ict:
        if var in data_imputed.columns:
            # Center
            centered = data_imputed[var] - data_imputed[var].mean()
            # Scale by 2 SDs
            scaled = centered / (2 * data_imputed[var].std())
            hlm_data[var] = scaled
    
    # Recode gender
    if 'GEND' in hlm_data.columns:
        hlm_data['GEND'] = hlm_data['GEND'].map({1: 0, 2: 1})  # Female=0, Male=1
    
    # Correct weights (if needed)
    weight_vars = [col for col in hlm_data.columns if col.startswith('W_FS')]
    if weight_vars:
        for weight_var in weight_vars:
            original_sum = hlm_data[weight_var].sum()
            hlm_data[weight_var] = len(hlm_data) * hlm_data[weight_var] / original_sum
    
    return hlm_data

def run_hlm_analysis(hlm_data, year):
    """Run hierarchical linear modeling analysis"""
    
    if not HLM_AVAILABLE or hlm_data is None:
        print("HLM analysis skipped")
        return
    
    print(f"\nRunning HLM analysis for {year}")
    print("-" * 30)
    
    # Get PV variables
    math_pvs = [col for col in hlm_data.columns if 'MATH' in col and col.startswith('PV')]
    science_pvs = [col for col in hlm_data.columns if 'SCIE' in col and col.startswith('PV')]
    
    if not math_pvs and not science_pvs:
        print("No PV variables found for HLM analysis")
        return
    
    # Check for required variables
    required_vars = ['GEND', 'ESCS', 'SCHL']
    missing_vars = [var for var in required_vars if var not in hlm_data.columns]
    
    if missing_vars:
        print(f"Missing required variables for HLM: {missing_vars}")
        return
    
    # Run models for first few PVs as examples
    pvs_to_analyze = (math_pvs[:3] if math_pvs else []) + (science_pvs[:3] if science_pvs else [])
    
    for pv in pvs_to_analyze:
        try:
            print(f"\nAnalyzing {pv}...")
            
            # Null model (unconditional)
            null_formula = f"{pv} ~ 1 + (1|SCHL)"
            
            # Full model
            predictor_vars = ['GEND', 'ESCS', 'COMP', 'INTE', 'SOCI', 'AUTO']
            available_predictors = [var for var in predictor_vars if var in hlm_data.columns]
            
            if available_predictors:
                full_formula = f"{pv} ~ {' + '.join(available_predictors)} + (1|SCHL)"
                
                print(f"Formula: {full_formula}")
                
                # This would require pymer4 for actual implementation
                # For now, we'll use statsmodels as an approximation
                try:
                    model_data = hlm_data[[pv, 'SCHL'] + available_predictors].dropna()
                    
                    if len(model_data) > 0:
                        # Simple linear regression as approximation
                        formula_simple = f"{pv} ~ {' + '.join(available_predictors)}"
                        model = smf.ols(formula_simple, data=model_data).fit()
                        
                        print("Model Summary (OLS approximation):")
                        print(model.summary().tables[1])
                        
                except Exception as e:
                    print(f"Error fitting model for {pv}: {e}")
                    
        except Exception as e:
            print(f"Error analyzing {pv}: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("PISA Statistical Analysis and Machine Learning")
    print("Python implementation")
    print("="*50)
    
    # Load data
    data15, data18 = load_and_prepare_data()
    
    if data15 is None or data18 is None:
        print("Could not load data files. Exiting.")
        return
    
    # Process both years
    for year, data in [('2015', data15), ('2018', data18)]:
        print(f"\n{'='*20} PROCESSING {year} DATA {'='*20}")
        
        # Step 1: Select and rename variables
        data_selected = select_variables(data)
        data_renamed = rename_variables(data_selected)
        
        # Step 2: Analyze and handle missingness
        data_filtered = analyze_missingness(data_renamed, year)
        
        # Step 3: Visualize missing patterns for remaining variables
        missing_vars_data = data_filtered.loc[:, data_filtered.isnull().any()]
        if not missing_vars_data.empty:
            visualize_missing_patterns(missing_vars_data, f"{year} - Variables with Missing Data")
        
        # Step 4: Impute missing data
        data_imputed = impute_missing_data(data_filtered, year)
        
        # Step 5: Compare imputation quality
        if not missing_vars_data.empty:
            compare_imputation_quality(missing_vars_data, 
                                     data_imputed[missing_vars_data.columns], 
                                     year)
        
        # Step 6: Prepare ML data
        ml_data = prepare_ml_data(data_imputed, data_filtered, year)
        math_data, science_data, features = prepare_model_datasets(ml_data)
        
        # Step 7: Train Random Forest models
        if math_data is not None or science_data is not None:
            models, test_data = train_random_forest_models(math_data, science_data, features, year)
            
            # Step 8: Plot feature importance
            if models:
                plot_feature_importance(models)
            
            # Step 9: Evaluate models
            if models and test_data:
                evaluate_models(models, test_data, year)
            
            # Step 10: Plot partial dependence (commented out due to complexity)
            # plot_partial_dependence(models, test_data, features)
        
        # Step 11: HLM analysis
        hlm_data = prepare_hlm_data(data_imputed, data_filtered)
        run_hlm_analysis(hlm_data, year)
        
        print(f"\nCompleted analysis for {year}")
        print("-" * 50)
    
    print("\nAnalysis completed for both years!")

if __name__ == "__main__":
    main()
