import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder

import joblib
import io
import time
from scipy.stats import randint

start_time = time.time()

def load_vcf(file_path):
    with open(file_path, 'rt') as f:
        lines = [line for line in f if not line.startswith('##')]
        df = pd.read_csv(
            io.StringIO(''.join(lines)),
            sep='\t',
            comment='#',
            names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
        )
    return df


BRCA1_START = 43044295
BRCA1_END = 43125482


vcf_file = 'homo_sapiens-chr17.vcf'
df = load_vcf(vcf_file)


df_brca1 = df[(df['POS'] >= BRCA1_START) & (df['POS'] <= BRCA1_END)].copy()


def extract_clinvar_annotation(info):
    if 'CLIN_pathogenic' in info or 'CLIN_likely_pathogenic' in info:
        return 1
    elif 'CLIN_benign' in info or 'CLIN_likely_benign' in info:
        return 0
    else:
        return np.nan

df_brca1['label'] = df_brca1['INFO'].apply(extract_clinvar_annotation)
df_brca1 = df_brca1[['POS', 'REF', 'ALT', 'INFO', 'label']].rename(
    columns={'POS': 'pk', 'REF': 'rk', 'ALT': 'ak', 'INFO': 'Fk'}
)


df_brca1['pos_norm'] = (df_brca1['pk'] - BRCA1_START) / (BRCA1_END - BRCA1_START)
assert df_brca1['pos_norm'].min() >= 0 and df_brca1['pos_norm'].max() <= 1, "Normalization out of range"

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
rk_ak_encoded = encoder.fit_transform(df_brca1[['rk', 'ak']])
rk_ak_columns = encoder.get_feature_names_out(['rk', 'ak'])
rk_ak_df = pd.DataFrame(rk_ak_encoded, columns=rk_ak_columns, index=df_brca1.index)


df_brca1 = pd.concat([df_brca1, rk_ak_df], axis=1)
feature_columns = ['pos_norm'] + list(rk_ak_columns)
X = df_brca1[feature_columns]
y = df_brca1['label']
df_clean = df_brca1.dropna(subset=['pos_norm', 'rk', 'ak', 'label'])
# Статистика після відбору BRCA1 та очищення
total_records = len(df_clean)
pathogenic_count = (df_clean['label'] == 1).sum()
benign_count = (df_clean['label'] == 0).sum()

print("\nBRCA1 records statistics after filtering:")
print(f"Total records: {total_records}")
print(f"Pathogenic variants: {pathogenic_count}")
print(f"Benign variants: {benign_count}")

X_clean = df_clean[feature_columns]
y_clean = df_clean['label']

X_clean = X_clean.astype(float)

df_clean.to_csv('brca1_processed.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")


param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42
)
rf_model.fit(X_train, y_train)

best_rf_model = rf_model.best_estimator_

y_pred = best_rf_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test set metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Best parameters: {rf_model.best_params_}")


cv_scores = cross_val_score(best_rf_model, X_clean, y_clean, cv=5, scoring='accuracy')
print("\nCross-validation results:")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

joblib.dump(best_rf_model, 'rf_brca1_model.pkl')
