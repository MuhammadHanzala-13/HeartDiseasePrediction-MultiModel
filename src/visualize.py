import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set global style for professional look
sns.set_style("white")
plt.rcParams.update({
    'figure.autolayout': True,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# Custom Color Palette
PRIMARY_COLOR = "#2563EB"  # Professional Blue
SECONDARY_COLOR = "#DC2626" # Alert Red
ACCENT_COLOR = "#059669"    # Success Green

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                cbar=False, linewidths=2, linecolor='white', 
                annot_kws={"size": 16, "weight": "bold"}, square=True)
    
    ax.set_xlabel("Predicted Condition", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("Actual Condition", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=15, pad=20, fontweight='bold')
    
    ax.set_xticklabels(["Healthy", "Disease"], fontsize=10)
    ax.set_yticklabels(["Healthy", "Disease"], fontsize=10, verticalalignment='center')
    
    return fig

def plot_roc_curve(y_true, y_score, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color=PRIMARY_COLOR, lw=3)
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", alpha=0.8)
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title(f"{model_name}\nROC Analysis", fontsize=15, pad=20, fontweight='bold')
    
    ax.legend(loc="lower right", frameon=True, fontsize=10, fancybox=True)
    ax.grid(True, linestyle=':', alpha=0.4)
    sns.despine()
    
    return fig

def plot_feature_importance(importances, feature_names, model_name="Random Forest"):
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Using a clean gradient palette
    colors = sns.color_palette("viridis", len(df_imp))
    sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, hue="Feature", legend=False, palette="viridis")
    
    ax.set_title(f"Key Predictors: {model_name}", fontsize=15, pad=20, fontweight='bold')
    ax.set_xlabel("Significance Score", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("Clinical Feature", fontsize=12, labelpad=10, fontweight='bold')
    
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 8})
    
    ax.set_title("Feature Correlation Heatmap", fontsize=15, pad=20, fontweight='bold')
    return fig

def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['target'].value_counts()
    
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=[ACCENT_COLOR, SECONDARY_COLOR], hue=counts.index, legend=False)
    
    ax.set_title("Disease Prevalence in Dataset", fontsize=15, pad=20, fontweight='bold')
    ax.set_xlabel("Condition (0: Healthy, 1: Diseased)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Patient Count", fontsize=12, fontweight='bold')
    ax.set_xticklabels(["Healthy", "Heart Disease"])
    
    sns.despine()
    return fig

def plot_thalach_vs_age(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target", 
                    palette={0: ACCENT_COLOR, 1: SECONDARY_COLOR}, 
                    alpha=0.7, s=100, ax=ax)
    
    ax.set_title("Max Heart Rate vs Age", fontsize=15, pad=20, fontweight='bold')
    ax.set_xlabel("Age", fontsize=12, fontweight='bold')
    ax.set_ylabel("Max Heart Rate (thalach)", fontsize=12, fontweight='bold')
    ax.legend(title="Condition", labels=["Healthy", "Disease"])
    
    sns.despine()
    return fig
