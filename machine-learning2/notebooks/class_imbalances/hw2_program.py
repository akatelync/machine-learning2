import time
import json
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(os.path.join("../../../data", 'ml2_student_performance.csv'))
# df = pd.concat([df[df["Class"] == 0].head(
#     100), df[df["Class"] == 1].head(50)])

X, y = df.drop("Class", axis=1), df["Class"]

(X_trainval, X_holdout, y_trainval, y_holdout) = train_test_split(X, y,
                                                                  random_state=143,
                                                                  test_size=0.25,
                                                                  stratify=y)

random_state = 42

param_grid = {
    "models": {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "LogisticRegressor": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        "SVC": SVC(random_state=random_state),
    },
    "samplers": {
        "ADASYN": ADASYN(random_state=random_state),
        "Random Oversampling": RandomOverSampler(random_state=random_state),
        "Random Undersampling": RandomUnderSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
        "Tomek Links": TomekLinks(),
        "SMOTE-Tomek": SMOTETomek(random_state=random_state)
    }
}


def evaluate_kfold_resampling(X_train, y_train, X_test, y_test, sampler, model):
    sampler = clone(sampler)
    model = clone(model)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_scores = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auprc = auc(recall, precision)
    rec = sensitivity_score(y_test, y_scores)
    gmean = geometric_mean_score(y_test, y_scores)
    return auprc, rec, gmean, precision, recall


def evaluate_fold(model_name, sampler_name, model, sampler, X_trainval, y_trainval, train_index, val_index):
    X_train = X_trainval.iloc[train_index]
    X_val = X_trainval.iloc[val_index]
    y_train = y_trainval.iloc[train_index]
    y_val = y_trainval.iloc[val_index]
    auprc, rec, gmean, precision, recall = evaluate_kfold_resampling(
        X_train, y_train, X_val, y_val, sampler, model)
    return model_name, sampler_name, auprc, rec, gmean, precision, recall


def compare_models_samplers_parallel(param_grid, X_trainval, y_trainval, n_splits=5, num_workers=10):
    skf = StratifiedKFold(n_splits=n_splits)
    tasks = []
    for model_name, model in param_grid["models"].items():
        for sampler_name, sampler in param_grid["samplers"].items():
            for train_index, val_index in skf.split(X_trainval, y_trainval):
                tasks.append((model_name, sampler_name, model,
                             sampler, train_index, val_index))
    results_dict = {}
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                evaluate_fold,
                model_name,
                sampler_name,
                model,
                sampler,
                X_trainval,
                y_trainval,
                train_index,
                val_index,
            )
            for model_name, sampler_name, model, sampler, train_index, val_index in tasks
        ]

        total_tasks = len(futures)
        counter = 0

        with tqdm(total=total_tasks, desc="Evaluating tasks") as pbar:
            for future in concurrent.futures.as_completed(futures):
                counter += 1
                model_name, sampler_name, auprc, rec, gmean, precision, recall = future.result()
                if model_name not in results_dict:
                    results_dict[model_name] = {}
                if sampler_name not in results_dict[model_name]:
                    results_dict[model_name][sampler_name] = {
                        "auprc": [], "rec": [], "gmean": [], "precision": [], "recall": [], }
                results_dict[model_name][sampler_name]["auprc"].append(auprc)
                results_dict[model_name][sampler_name]["rec"].append(rec)
                results_dict[model_name][sampler_name]["gmean"].append(gmean)
                results_dict[model_name][sampler_name]["precision"].append(
                    precision)
                results_dict[model_name][sampler_name]["recall"].append(recall)
                results_dict[model_name][sampler_name]["runtime"] = time.time(
                ) - start_time
                pbar.update(1)
                pbar.set_postfix_str(f"{counter}/{total_tasks}")

    for model_name in results_dict:
        for sampler_name in results_dict[model_name]:
            results_dict[model_name][sampler_name] = {
                "auprc": np.mean(results_dict[model_name][sampler_name]["auprc"]),
                "rec": np.mean(results_dict[model_name][sampler_name]["rec"]),
                "gmean": np.mean(results_dict[model_name][sampler_name]["gmean"]),
                "precision": np.mean(results_dict[model_name][sampler_name]["precision"]),
                "recall": np.mean(results_dict[model_name][sampler_name]["recall"]),
                "runtime": results_dict[model_name][sampler_name]["runtime"]
            }

    total_time = time.time() - start_time
    print(f"\nAll evaluations completed in {total_time:.2f} seconds!\n")
    return results_dict


if __name__ == "__main__":
    # Example call (replace with real data)
    results = compare_models_samplers_parallel(
        param_grid, X_trainval, y_trainval)

    with open("final_results.json", "w") as outfile:
        json.dump(results, outfile)
