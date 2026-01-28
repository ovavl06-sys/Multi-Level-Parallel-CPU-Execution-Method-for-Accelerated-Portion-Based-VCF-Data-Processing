import pandas as pd
import numpy as np
import time
import io
from datetime import datetime
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

BRCA1_START = 43044295
BRCA1_END = 43125482
VCF_FILE = "homo_sapiens-chr17.vcf"

CORES_LIST = [1, 2, 4, 8, 14]
N_RUNS = 5  # number of repeated measurements for statistics


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_vcf(file_path):

    with open(file_path, "rt") as f:
        lines = [l for l in f if not l.startswith("##")]
    df = pd.read_csv(
        io.StringIO("".join(lines)),
        sep="\t",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
        low_memory=False
    )
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df = df.dropna(subset=["POS"])
    return df

def load_vcf_chunks(file_path, chunksize=100000):

    reader = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        names=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
        low_memory=False,
        chunksize=chunksize
    )
    chunks = []
    for chunk in reader:
        chunk["POS"] = pd.to_numeric(chunk["POS"], errors="coerce")
        chunk = chunk.dropna(subset=["POS"])
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    return df

def extract_label(info):
    if "CLIN_pathogenic" in info or "CLIN_likely_pathogenic" in info:
        return 1
    if "CLIN_benign" in info or "CLIN_likely_benign" in info:
        return 0
    return np.nan

param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2"]
}

def run_single(core_count, seed):
    np.random.seed(seed)
    start_total = time.time()


    if core_count == 1:
        df = load_vcf(VCF_FILE)
    else:
        df = load_vcf_chunks(VCF_FILE, chunksize=100000)


    df = df[(df["POS"] >= BRCA1_START) & (df["POS"] <= BRCA1_END)].copy()
    df["label"] = df["INFO"].apply(extract_label)
    df = df.dropna(subset=["label", "REF", "ALT"])


    df["pos_norm"] = (df["POS"] - BRCA1_START) / (BRCA1_END - BRCA1_START)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[["REF", "ALT"]])
    encoded_cols = encoder.get_feature_names_out(["REF", "ALT"])
    X = pd.concat(
        [df[["pos_norm"]].reset_index(drop=True),
         pd.DataFrame(encoded, columns=encoded_cols)],
        axis=1
    )
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    start_search = time.time()
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=core_count),
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="f1",
        n_jobs=core_count,
        random_state=seed
    )
    search.fit(X_train, y_train)
    search_time = time.time() - start_search


    best_model = search.best_estimator_
    start_cv = time.time()
    _ = cross_val_score(best_model, X_train, y_train, cv=5, n_jobs=core_count)
    cv_time = time.time() - start_cv

    _ = best_model.predict(X_test)

    total_time = time.time() - start_total

    return total_time, search_time, cv_time

results = {}
experiment_start = time.time()

for cores in CORES_LIST:
    log(f"Starting configuration: {cores} core(s)")
    total_times, search_times, cv_times = [], [], []

    for i in range(N_RUNS):
        log(f"  Run {i + 1}/{N_RUNS}")
        t_total, t_search, t_cv = run_single(core_count=cores, seed=100 + i)
        log(f"    Run time: {t_total:.2f}s (search: {t_search:.2f}s, CV: {t_cv:.2f}s)")
        total_times.append(t_total)
        search_times.append(t_search)
        cv_times.append(t_cv)

    results[cores] = {
        "total": np.array(total_times),
        "search": np.array(search_times),
        "cv": np.array(cv_times)
    }
    log(f"Finished configuration: {cores} core(s)")

log(f"Total experiment time: {(time.time() - experiment_start) / 60:.1f} minutes")


def summarize(arr):
    mean = np.mean(arr)
    sd = np.std(arr, ddof=1)
    cv = (sd / mean) * 100
    ci = stats.t.interval(
        0.95,
        len(arr) - 1,
        loc=mean,
        scale=sd / np.sqrt(len(arr))
    )
    return mean, sd, cv, ci


rows = []
for cores in CORES_LIST:
    mean, sd, cv, ci = summarize(results[cores]["total"])
    rows.append([
        "Sequential (1 core)" if cores == 1 else f"Parallel ({cores} cores)",
        round(mean, 2), round(sd, 2), round(cv, 2),
        f"[{ci[0]:.2f}, {ci[1]:.2f}]"
    ])

table = pd.DataFrame(
    rows, columns=["Configuration", "Mean (s)", "SD (s)", "CV (%)", "95% CI (s)"]
)
print("\nTable X. Statistical summary of runtime measurements (5 runs)")
print(table.to_string(index=False))


print("\nHyperparameter search cost (mean ± SD):")
for cores in CORES_LIST:
    mean_s, sd_s, _, _ = summarize(results[cores]["search"])
    label = "Sequential (1 core)" if cores == 1 else f"Parallel ({cores} cores)"
    print(f"{label}: {mean_s:.2f} ± {sd_s:.2f} s")

print("\nCross-validation cost (mean ± SD):")
for cores in CORES_LIST:
    mean_cv, sd_cv, _, _ = summarize(results[cores]["cv"])
    label = "Sequential (1 core)" if cores == 1 else f"Parallel ({cores} cores)"
    print(f"{label}: {mean_cv:.2f} ± {sd_cv:.2f} s")


baseline = results[1]["total"]
print("\nPaired t-test vs sequential:")
for cores in CORES_LIST:
    if cores == 1:
        continue
    t_stat, p_val = stats.ttest_rel(baseline, results[cores]["total"])
    print(f"{cores} cores: p-value = {p_val:.4e}")
