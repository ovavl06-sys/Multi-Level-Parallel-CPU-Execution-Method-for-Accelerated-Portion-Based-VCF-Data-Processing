import pandas as pd
import numpy as np
import time
import logging
import multiprocessing as mp

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint

from numba import jit, prange

N_CORES = 14
CHUNK_SIZE = 100_000

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start_time_total = time.time()

def load_vcf(file_path):
    with open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                header = line.lstrip('#').strip().split('\t')
                break
        df = pd.read_csv(f, sep='\t', names=header)
    return df



GENES = {
    'MLH1': {
        'start': 36993313,
        'end': 37050477,
        'vcf': r'homo_sapiens-chr3.vcf'
    },
    'TP53': {
        'start': 7668421,
        'end': 7687490,
        'vcf': r'homo_sapiens-chr17.vcf'
    },
    'CDKN2A': {
        'start': 21967751,
        'end': 21995301,
        'vcf': r'homo_sapiens-chr9.vcf'
    }
}

def extract_clinvar_annotation(info):
    info = str(info)
    if 'CLIN_pathogenic' in info or 'CLIN_likely_pathogenic' in info:
        return 1
    elif 'CLIN_benign' in info or 'CLIN_likely_benign' in info:
        return 0
    return np.nan


@jit(nopython=True, parallel=True)
def normalize_positions_parallel(pos_array, start, end):
    result = np.empty(len(pos_array))
    for i in prange(len(pos_array)):
        result[i] = (pos_array[i] - start) / (end - start)
    return result

def sequential_pipeline(gene, params):
    print(f"\n--- Sequential Processing: {gene} ---")
    start_time = time.time()

    df = load_vcf(params['vcf'])
    df = df[(df['POS'] >= params['start']) & (df['POS'] <= params['end'])].copy()
    df['label'] = df['INFO'].apply(extract_clinvar_annotation)
    df.dropna(subset=['label'], inplace=True)

    df['pos_norm'] = (df['POS'] - params['start']) / (params['end'] - params['start'])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['REF', 'ALT']])

    X = np.hstack([df[['pos_norm']].values, encoded])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions={
            'n_estimators': randint(50, 300),
            'max_depth': [None, 10, 20],
            'min_samples_split': randint(2, 20),
            'class_weight': ['balanced']
        },
        n_iter=30,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.best_estimator_.predict(X_test)

    elapsed = time.time() - start_time

    metrics = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"Time: {elapsed:.2f}s")
    print(metrics)

    return metrics


def parallel_pipeline(gene, params):
    print(f"\n--- Parallel Processing: {gene} ({N_CORES} cores) ---")
    start_time = time.time()

    chunks = []
    for chunk in pd.read_csv(
        params['vcf'],
        sep='\t',
        comment='#',
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'],
        chunksize=CHUNK_SIZE
    ):
        filtered = chunk[(chunk['POS'] >= params['start']) & (chunk['POS'] <= params['end'])]
        if not filtered.empty:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    df['label'] = df['INFO'].apply(extract_clinvar_annotation)
    df.dropna(subset=['label'], inplace=True)

    df['pos_norm'] = normalize_positions_parallel(
        df['POS'].values.astype(np.float64),
        params['start'], params['end']
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['REF', 'ALT']])

    X = np.hstack([df[['pos_norm']].values, encoded])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=N_CORES
        ),
        param_distributions={
            'n_estimators': randint(50, 300),
            'max_depth': [None, 20]
        },
        n_iter=20,
        cv=5,
        scoring='f1',
        n_jobs=N_CORES
    )

    rf.fit(X_train, y_train)
    y_pred = rf.best_estimator_.predict(X_test)

    elapsed = time.time() - start_time

    metrics = {
        'time': elapsed,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"Time: {elapsed:.2f}s")
    print(metrics)

    return metrics


if __name__ == '__main__':

    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Parallel execution uses: {N_CORES} cores\n")

    for gene, params in GENES.items():
        seq = sequential_pipeline(gene, params)
        par = parallel_pipeline(gene, params)

        speedup = seq['time'] / par['time']

        print(f"\n>>> SPEEDUP for {gene}: {speedup:.2f}×")
        print("=" * 60)

    print(f"\nTotal runtime: {time.time() - start_time_total:.2f}s")
