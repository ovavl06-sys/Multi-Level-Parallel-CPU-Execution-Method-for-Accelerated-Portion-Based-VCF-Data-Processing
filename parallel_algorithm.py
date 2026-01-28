import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from numba import jit, prange
import multiprocessing as mp
import joblib
import time
import importlib.util
import sys
import traceback
import logging
from scipy.stats import randint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BRCA1_START = 43044295
BRCA1_END = 43125482
VCF_PATH = 'homo_sapiens-chr17.vcf'
SEQUENTIAL_SCRIPT = 'sequental_algorithm.py'
CHUNK_SIZE = 100000

def load_vcf(file_path, start, end):
    chunks = []
    try:
        logging.info(f"Reading VCF file in chunks: {file_path}")
        for chunk in pd.read_csv(file_path, sep='\t', comment='#',
                                 names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'],
                                 chunksize=CHUNK_SIZE):
            chunk_filtered = chunk[(chunk['POS'] >= start) & (chunk['POS'] <= end)].copy()
            if not chunk_filtered.empty:
                chunks.append(chunk_filtered)
        if not chunks:
            logging.error("No records found in the specified region.")
            sys.exit(1)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except Exception as e:
        logging.error(f"Error reading VCF: {e}")
        sys.exit(1)

def extract_label(info):
    if 'CLIN_pathogenic' in info or 'CLIN_likely_pathogenic' in info:
        return 1
    elif 'CLIN_benign' in info or 'CLIN_likely_benign' in info:
        return 0
    else:
        return np.nan

def process_block(df_block):
    try:
        df_block = df_block[(df_block['POS'] >= BRCA1_START) & (df_block['POS'] <= BRCA1_END)].copy()
        return df_block
    except Exception as e:
        logging.error(f"Error in block processing: {e}")
        return pd.DataFrame()

def annotate_task(df, queue):
    try:
        logging.info("Running annotation task")
        df['label'] = df['INFO'].apply(extract_label)
        df = df[['POS', 'REF', 'ALT', 'INFO', 'label']].rename(
            columns={'POS': 'pk', 'REF': 'rk', 'ALT': 'ak', 'INFO': 'Fk'})
        queue.put(df)
    except Exception as e:
        logging.error(f"Error in annotation task: {e}")
        queue.put(None)

@jit(nopython=True, parallel=True)
def normalize_positions(pos_array, start, end):
    result = np.empty(len(pos_array))
    for i in prange(len(pos_array)):
        result[i] = (pos_array[i] - start) / (end - start)
    return result

def normalize_task(df, queue):
    try:
        logging.info("Running normalization task")
        pos_norm = normalize_positions(df['pk'].values, BRCA1_START, BRCA1_END)
        queue.put(pd.Series(pos_norm, index=df.index, name='pos_norm'))
    except Exception as e:
        logging.error(f"Error in normalization task: {e}")
        queue.put(None)

def encode_task(df, queue):
    try:
        logging.info("Running encoding task")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        rk_ak_encoded = encoder.fit_transform(df[['rk', 'ak']])
        rk_ak_columns = encoder.get_feature_names_out(['rk', 'ak'])
        rk_ak_df = pd.DataFrame(rk_ak_encoded, columns=rk_ak_columns, index=df.index)
        queue.put((rk_ak_df, rk_ak_columns))
    except Exception as e:
        logging.error(f"Error in encoding task: {e}")
        queue.put(None)

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.05):
    """Обчислення 95% довірчого інтервалу для метрики через бутстреп"""
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), n, replace=True)
        scores.append(metric_func(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, 100*alpha/2)
    upper = np.percentile(scores, 100*(1-alpha/2))
    return lower, upper

def run_hybrid_pipeline(n_cores):
    start_time = time.time()

    # --- VCF ---
    df = load_vcf(VCF_PATH, BRCA1_START, BRCA1_END)
    blocks = np.array_split(df, n_cores)
    with mp.Pool(n_cores) as pool:
        processed_blocks = pool.map(process_block, blocks)
    df_brca1 = pd.concat([b for b in processed_blocks if not b.empty], ignore_index=True)
    df_brca1.dropna(subset=['POS', 'REF', 'ALT', 'INFO'], inplace=True)

    manager = mp.Manager()
    annotate_queue = manager.Queue()
    annotate_process = mp.Process(target=annotate_task, args=(df_brca1, annotate_queue))
    annotate_process.start()
    annotate_process.join()
    df_annotated = annotate_queue.get()
    if df_annotated is None:
        logging.error("Annotation failed")
        sys.exit(1)

    normalize_queue = manager.Queue()
    encode_queue = manager.Queue()
    normalize_process = mp.Process(target=normalize_task, args=(df_annotated, normalize_queue))
    encode_process = mp.Process(target=encode_task, args=(df_annotated, encode_queue))
    normalize_process.start()
    encode_process.start()
    normalize_process.join()
    encode_process.join()
    pos_norm = normalize_queue.get()
    encode_result = encode_queue.get()
    if pos_norm is None or encode_result is None:
        logging.error("Normalization or encoding failed")
        sys.exit(1)
    rk_ak_df, rk_ak_columns = encode_result

    df_brca1 = df_annotated.copy()
    df_brca1['pos_norm'] = pos_norm
    df_brca1 = pd.concat([df_brca1, rk_ak_df], axis=1)
    df_brca1.dropna(subset=['label', 'pos_norm'], inplace=True)

    feature_columns = ['pos_norm'] + list(rk_ak_columns)
    X = df_brca1[feature_columns].astype(float)
    y = df_brca1['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # --- RandomizedSearchCV ---
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2']
    }

    model = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    y_pred = best_model.predict(X_test)

    end_time = time.time()
    elapsed = end_time - start_time

    # --- Метрики ---
    metrics = {}
    metrics['execution_time'] = elapsed
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)

    # 95% CI
    y_test_arr = y_test.values
    y_pred_arr = y_pred
    metrics['accuracy_ci'] = bootstrap_ci(y_test_arr, y_pred_arr, accuracy_score)
    metrics['precision_ci'] = bootstrap_ci(y_test_arr, y_pred_arr, precision_score)
    metrics['recall_ci'] = bootstrap_ci(y_test_arr, y_pred_arr, recall_score)
    metrics['f1_ci'] = bootstrap_ci(y_test_arr, y_pred_arr, f1_score)

    return metrics

def run_sequential_script():
    try:
        spec = importlib.util.spec_from_file_location("sequental_algorithm", SEQUENTIAL_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'elapsed_time', None)
    except Exception as e:
        logging.error(f"Error running sequential script: {e}")
        return None

def compare_results():
    logging.info("Отримуємо час виконання послідовного алгоритму...")
    sequential_time = run_sequential_script()
    if sequential_time:
        print(f"Послідовний час виконання: {sequential_time:.2f} сек\n")
    else:
        print("Не вдалося отримати час послідовного алгоритму\n")

    for cores in [2, 4, 8, 14]:
        logging.info(f"Запуск гібридного алгоритму на {cores} ядрах...")
        try:
            metrics = run_hybrid_pipeline(cores)
            speedup = sequential_time / metrics['execution_time'] if sequential_time else None
            efficiency = speedup / cores if speedup else None

            print(f"\n=== {cores} ядер ===")
            print(f"Час виконання: {metrics['execution_time']:.2f} сек")
            if speedup: print(f"Прискорення (Speedup): {speedup:.2f}")
            if efficiency: print(f"Ефективність (Efficiency): {efficiency:.2f}")
            print(f"Accuracy: {metrics['accuracy']:.4f} (95% CI: {metrics['accuracy_ci']})")
            print(f"Precision: {metrics['precision']:.4f} (95% CI: {metrics['precision_ci']})")
            print(f"Recall: {metrics['recall']:.4f} (95% CI: {metrics['recall_ci']})")
            print(f"F1-score: {metrics['f1']:.4f} (95% CI: {metrics['f1_ci']})")
        except Exception as e:
            logging.error(f"Error for {cores} cores: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    try:
        compare_results()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
