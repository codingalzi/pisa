#%% [markdown]
# Libraries
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# %%
# Data manipulation and visualization
import pyreadstat  # for reading SPSS files
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %%
# Missing data handling
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#%%
# Imbalanced learning
from imblearn.over_sampling import RandomOverSampler

# Statistical models
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import scipy.stats as stats

# Visualization
from mpl_toolkits.mplot3d import Axes3D

# Select German samples from PISA 2015 and 2018
# PISA 2015, Student questionnaire SPSS
try:
    alldata15, meta15 = pyreadstat.read_sav("CY6_MS_CMB_STU_QQQ.sav")
    DEUdata15 = alldata15[alldata15['CNT'] == "DEU"].copy()
    del alldata15
except FileNotFoundError:
    print("PISA 2015 file not found. Please download from OECD website.")

# PISA 2018, Student questionnaire SPSS
try:
    alldata18, meta18 = pyreadstat.read_sav("CY07_MSU_STU_QQQ.sav")
    DEUdata18 = alldata18[alldata18['CNT'] == "DEU"].copy()
    del alldata18
except FileNotFoundError:
    print("PISA 2018 file not found. Please download from OECD website.")

# Select variables: demographics, ICT, maths, science, and weights
def select_variables(data):
    # Get PV columns for MATH and SCIE
    pv_math_cols = [col for col in data.columns if col.startswith('PV') and col.endswith('MATH')]
    pv_scie_cols = [col for col in data.columns if col.startswith('PV') and col.endswith('SCIE')]
    w_fs_cols = [col for col in data.columns if col.startswith('W_FS')]
    
    selected_cols = [
        "CNTSCHID", "ST004D01T", "ESCS", "INTICT", "COMPICT", 
        "AUTICT", "SOIAICT"
    ] + pv_math_cols + pv_scie_cols + w_fs_cols
    
    return data[selected_cols].copy()

dataA15 = select_variables(DEUdata15)
dataA18 = select_variables(DEUdata18)

# Change column names
rename_dict = {
    "CNTSCHID": "SCHL",
    "ST004D01T": "GEND", 
    "INTICT": "INTE",
    "COMPICT": "COMP",
    "AUTICT": "AUTO",
    "SOIAICT": "SOCI"
}

dataA15 = dataA15.rename(columns=rename_dict)
dataA18 = dataA18.rename(columns=rename_dict)

## Missing Data Analysis
# Find and remove rows with 100% ICT attitudes missing
ict_cols = ['INTE', 'COMP', 'AUTO', 'SOCI']

# For 2015
na_rows_ict15 = dataA15[ict_cols].isnull().sum(axis=1) / len(ict_cols)
print(f"Percentage of rows with 100% ICT missing (2015): {(na_rows_ict15 == 1).sum() / len(dataA15):.3f}")

# For 2018  
na_rows_ict18 = dataA18[ict_cols].isnull().sum(axis=1) / len(ict_cols)
print(f"Percentage of rows with 100% ICT missing (2018): {(na_rows_ict18 == 1).sum() / len(dataA18):.3f}")

# Remove rows with 100% ICT missingness
dataB15 = dataA15[na_rows_ict15 < 1].copy()
dataB18 = dataA18[na_rows_ict18 < 1].copy()

# Compare removed and remaining rows
dataRem15 = dataA15[na_rows_ict15 == 1].copy()
dataRem18 = dataA18[na_rows_ict18 == 1].copy()

# Gender distribution in removed data
print("2015 - Removed data gender distribution:")
print(f"Female: {(dataRem15['GEND'] == 1).sum()}")
print(f"Male: {(dataRem15['GEND'] == 2).sum()}")

# ESCS histograms comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2015
axes[0].hist(dataRem15['ESCS'].dropna(), alpha=0.5, color='red', label='Removed', bins=30)
axes[0].hist(dataB15['ESCS'].dropna(), alpha=0.5, color='blue', label='Remaining', bins=30)
axes[0].set_title('ESCS Distribution 2015')
axes[0].legend()

# 2018
axes[1].hist(dataRem18['ESCS'].dropna(), alpha=0.5, color='red', label='Removed', bins=30)
axes[1].hist(dataB18['ESCS'].dropna(), alpha=0.5, color='blue', label='Remaining', bins=30)
axes[1].set_title('ESCS Distribution 2018')
axes[1].legend()

plt.tight_layout()
plt.show()

# Missingness per variable
na_var15 = dataB15.isnull().sum() / len(dataB15)
na_var18 = dataB18.isnull().sum() / len(dataB18)

print("Missingness per variable (2015):")
print(na_var15[na_var15 > 0])

# Descriptives
print(f"\n2015 Descriptives:")
print(f"N schools: {dataB15['SCHL'].nunique()}")
print(f"Female students: {(dataB15['GEND'] == 1).sum() / len(dataB15) * 100:.1f}%")
print(f"Male students: {(dataB15['GEND'] == 2).sum() / len(dataB15) * 100:.1f}%")

print(f"\n2018 Descriptives:")
print(f"N schools: {dataB18['SCHL'].nunique()}")
print(f"Female students: {(dataB18['GEND'] == 1).sum() / len(dataB18) * 100:.1f}%")
print(f"Male students: {(dataB18['GEND'] == 2).sum() / len(dataB18) * 100:.1f}%")

# Subset with missing data for imputation
dataC15 = dataB15.loc[:, na_var15 > 0].copy()
dataC18 = dataB18.loc[:, na_var18 > 0].copy()

print(f"Percent missing in subset (2015): {dataC15.isnull().sum().sum() / dataC15.size:.3f}")
print(f"Percent missing in subset (2018): {dataC18.isnull().sum().sum() / dataC18.size:.3f}")

# Impute with IterativeImputer (similar to missForest)
np.random.seed(100)
imputer15 = IterativeImputer(random_state=100, max_iter=10)
dataI15 = pd.DataFrame(imputer15.fit_transform(dataC15), 
                       columns=dataC15.columns, 
                       index=dataC15.index)

np.random.seed(100)
imputer18 = IterativeImputer(random_state=100, max_iter=10)
dataI18 = pd.DataFrame(imputer18.fit_transform(dataC18), 
                       columns=dataC18.columns, 
                       index=dataC18.index)

# Compare histograms before/after imputation
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Before/After Imputation Comparison')

for i, col in enumerate(ict_cols + ['ESCS']):
    # 2015
    axes[0, i].hist(dataC15[col].dropna(), alpha=0.5, color='red', label='Original', bins=30)
    axes[0, i].hist(dataI15[col], alpha=0.5, color='green', label='Imputed', bins=30)
    axes[0, i].set_title(f'{col} 2015')
    axes[0, i].legend()
    
    # 2018
    axes[1, i].hist(dataC18[col].dropna(), alpha=0.5, color='red', label='Original', bins=30)
    axes[1, i].hist(dataI18[col], alpha=0.5, color='green', label='Imputed', bins=30)
    axes[1, i].set_title(f'{col} 2018')
    axes[1, i].legend()

plt.tight_layout()
plt.show()

# Prepare data for machine learning
def prepare_ml_data(dataB, dataI, year):
    # Get PV columns
    pv_cols = [col for col in dataB.columns if col.startswith('PV')]
    dataPV = dataB[['GEND'] + pv_cols].copy()
    
    # Combine imputed and PV data
    dataD = pd.concat([dataI, dataPV], axis=1)
    
    # Recode gender as 0/1 factor
    dataD['GEND'] = dataD['GEND'].map({1: 0, 2: 1}).astype('category')
    
    # Calculate mean scores
    math_cols = [col for col in dataD.columns if 'MATH' in col]
    scie_cols = [col for col in dataD.columns if 'SCIE' in col]
    
    dataD['meansM'] = dataD[math_cols].mean(axis=1)
    dataD['meansS'] = dataD[scie_cols].mean(axis=1)
    
    # Create proficiency levels
    dataD['Sc'] = pd.cut(dataD['meansS'], 
                        bins=[0, 409.54, 633.33, 1000], 
                        labels=['1', '2', '3'])
    dataD['M'] = pd.cut(dataD['meansM'], 
                       bins=[0, 420.07, 606.99, 1000], 
                       labels=['1', '2', '3'])
    
    # Select final variables
    dataSc = dataD[['INTE', 'COMP', 'AUTO', 'SOCI', 'ESCS', 'GEND', 'Sc']].copy()
    dataM = dataD[['INTE', 'COMP', 'AUTO', 'SOCI', 'ESCS', 'GEND', 'M']].copy()
    
    return dataSc, dataM

dataSc15, dataM15 = prepare_ml_data(dataB15, dataI15, 2015)
dataSc18, dataM18 = prepare_ml_data(dataB18, dataI18, 2018)

# Check class imbalance
print("Class distribution 2015:")
print("Math:", dataM15['M'].value_counts().sort_index())
print("Science:", dataSc15['Sc'].value_counts().sort_index())

print("\nClass distribution 2018:")
print("Math:", dataM18['M'].value_counts().sort_index())
print("Science:", dataSc18['Sc'].value_counts().sort_index())

# Split 2015 into train/test sets
np.random.seed(100)
X_sc15 = dataSc15.drop('Sc', axis=1)
y_sc15 = dataSc15['Sc']
X_train_sc15, X_test_sc15, y_train_sc15, y_test_sc15 = train_test_split(
    X_sc15, y_sc15, test_size=0.2, random_state=100, stratify=y_sc15)

np.random.seed(100)
X_m15 = dataM15.drop('M', axis=1)
y_m15 = dataM15['M']
X_train_m15, X_test_m15, y_train_m15, y_test_m15 = train_test_split(
    X_m15, y_m15, test_size=0.2, random_state=100, stratify=y_m15)

# Oversample training sets
ros = RandomOverSampler(random_state=100)
X_train_sc15_over, y_train_sc15_over = ros.fit_resample(X_train_sc15, y_train_sc15)
X_train_m15_over, y_train_m15_over = ros.fit_resample(X_train_m15, y_train_m15)

print("After oversampling:")
print("Math training:", pd.Series(y_train_m15_over).value_counts().sort_index())
print("Science training:", pd.Series(y_train_sc15_over).value_counts().sort_index())

# Decision Tree visualization
dt_model = DecisionTreeClassifier(random_state=100, max_depth=4)
dt_model.fit(X_train_m15_over, y_train_m15_over)

plt.figure(figsize=(15, 10))
plot_tree(dt_model, feature_names=X_train_m15_over.columns, 
          class_names=['1', '2', '3'], filled=True)
plt.title('Decision Tree for Mathematics')
plt.show()

# Random Forest models
np.random.seed(100)
rf_m = RandomForestClassifier(n_estimators=100, random_state=100)
rf_m.fit(X_train_m15_over, y_train_m15_over)

np.random.seed(100)
rf_sc = RandomForestClassifier(n_estimators=100, random_state=100)
rf_sc.fit(X_train_sc15_over, y_train_sc15_over)

# Feature importance plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Math importance
importances_m = rf_m.feature_importances_
axes[0].barh(X_train_m15_over.columns, importances_m)
axes[0].set_title('Mathematics Feature Importance')

# Science importance
importances_sc = rf_sc.feature_importances_
axes[1].barh(X_train_sc15_over.columns, importances_sc)
axes[1].set_title('Science Feature Importance')

plt.tight_layout()
plt.show()

# Model evaluation
# Predictions
pred_m15 = rf_m.predict(X_test_m15)
pred_m18 = rf_m.predict(dataM18.drop('M', axis=1))
pred_sc15 = rf_sc.predict(X_test_sc15)  
pred_sc18 = rf_sc.predict(dataSc18.drop('Sc', axis=1))

# Classification reports
print("Mathematics 2015 Test Set:")
print(classification_report(y_test_m15, pred_m15))

print("\nScience 2015 Test Set:")
print(classification_report(y_test_sc15, pred_sc15))

# ROC curves for multiclass (one-vs-rest approach)
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc(y_true, y_pred_proba, classes, title):
    # Binarize the output
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    
    plt.figure(figsize=(8, 6))
    
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {title}')
    plt.legend()
    plt.show()

# Get prediction probabilities
pred_proba_m15 = rf_m.predict_proba(X_test_m15)
pred_proba_sc15 = rf_sc.predict_proba(X_test_sc15)

# Plot ROC curves
plot_multiclass_roc(y_test_m15, pred_proba_m15, ['1', '2', '3'], 'Mathematics 2015')
plot_multiclass_roc(y_test_sc15, pred_proba_sc15, ['1', '2', '3'], 'Science 2015')

# Partial dependence plots (simplified version)
from sklearn.inspection import partial_dependence, plot_partial_dependence

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Partial Dependence Plots')

features = ['INTE', 'COMP', 'AUTO', 'SOCI', 'ESCS']

# Mathematics model - Class 1 vs 3
for i, feature in enumerate(features):
    # Class 1 (low proficiency)
    pd_result = partial_dependence(rf_m, X_train_m15_over, [i], 
                                  kind='average')
    axes[0, i].plot(pd_result['values'][0], pd_result['average'][0])
    axes[0, i].set_title(f'Math Class 1 - {feature}')
    axes[0, i].set_ylim(0, 0.5)

# Science model - Class 1
for i, feature in enumerate(features):
    pd_result = partial_dependence(rf_sc, X_train_sc15_over, [i], 
                                  kind='average')
    axes[1, i].plot(pd_result['values'][0], pd_result['average'][0])
    axes[1, i].set_title(f'Science Class 1 - {feature}')
    axes[1, i].set_ylim(0, 0.5)

plt.tight_layout()
plt.show()

print("Machine Learning analysis completed!")
print("\nFor the statistical multilevel modeling part, you would need to:")
print("1. Use statsmodels.formula.api.mixedlm for hierarchical linear models")
print("2. Implement proper survey weights handling")
print("3. Calculate plausible values statistics properly")
print("4. This requires more complex implementation beyond this basic conversion")
