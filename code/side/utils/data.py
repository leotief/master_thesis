# the data of the experiments and demos, and copies of it with missing values
#
# the data itself is always complete, missingness is added afterwards by setting
# entries to nan. everything else in the code uses the same convention: a nan
# means missing, and obs_mask = ~torch.isnan(x) gives the observed features.
# features are dropped completely at random (MCAR), independent of their value.
#   classification  sklearn's make_classification, split into train and test
#   hide            a copy of points with a fixed number of features per point set to nan
#                   (for the single test point of the demo figures)
#   hide_entries    a copy of points with every entry set to nan with some probability
#   load_data       classification and hide_entries, for train.py and estimate.py, as tensors
#   load_real       a real dataset (sklearn or openml), split, hidden by hide_entries
#   loader          a shuffled loader over points, to fit a density model

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split

# make_classification (gaussian clusters, balanced classes) with features
# features, split into train and test (stratified, so both have the same class
# balance). informative is the fraction of the features that carry the classes
# (at least 2), redundant the fraction that are linear combinations of those, the
# others are noise. None keeps
# sklearn's default: 2 informative, 2 redundant (linear combinations of those),
# the rest noise. standardize scales every feature to mean 0 and variance 1 on
# the training points (the informative features of make_classification have a
# variance that grows with their number). returns x_train, x_test, y_train,
# y_test as numpy arrays
def classification(features, informative=None, samples=10_000, test_size=0.2, seed=42, standardize=False, redundant=0.0):
    shape = {} if informative is None else dict(n_informative=max(2, round(informative * features)), n_redundant=round(redundant * features))
    x, y = make_classification(n_samples=samples, n_features=features, weights=[0.5, 0.5], random_state=seed, **shape)
    x_train, x_test, y_train, y_test = train_test_split(x.astype(np.float32), y.astype(np.int64), test_size=test_size,
                                                        random_state=seed, stratify=y)
    if standardize:
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    return x_train, x_test, y_train, y_test


# a copy of the point x (D,) or the points x (B, D) (numpy) with int(missing_rate
# * D) of the features of every point set to nan, chosen at random by rng for
# every point
def hide(x, missing_rate, rng):
    x = x.copy()
    n_hidden = int(missing_rate * x.shape[-1])
    for row in x.reshape(-1, x.shape[-1]):
        row[rng.choice(len(row), n_hidden, replace=False)] = np.nan
    return x


# a copy of the points x (B, D) (numpy) with every entry set to nan independently with
# probability missing_rate (mcar per entry): on average that fraction is missing, some
# points miss more, some none. used by all experiments: a fixed number per point would be
# 0 at low rates for few features
def hide_entries(x, missing_rate, rng):
    x = x.copy()
    x[rng.random(x.shape) < missing_rate] = np.nan
    return x


# the classification data with n_features features, informative of them
# informative and the others noise, samples points of which a fraction test_size
# are test points (by default 8000 training and 2000 test points), standardized
# or not, every entry hidden with probability missing_rate (hide_entries). the
# points are the same for every seed, which features are hidden is not. returns the training points (complete and incomplete), their labels, the
# test points (complete and incomplete) and their labels, as tensors
def load_data(n_features, missing_rate, informative, seed, device, standardize=True, samples=10_000, test_size=0.2, redundant=0.0):
    x_train, x_test, y_train, y_test = classification(n_features, informative, samples, test_size, standardize=standardize,
                                                      redundant=redundant)
    rng = np.random.default_rng(seed)
    x_missing = hide_entries(x_train, missing_rate, rng)
    x_test_missing = hide_entries(x_test, missing_rate, rng)
    arrays = (x_train, x_missing, y_train, x_test, x_test_missing, y_test)
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]


# a shuffled loader over the points x (tensor), what the density models fit on
def loader(x, batch_size=64):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=batch_size, shuffle=True)


# the real datasets: complete, continuous features and a binary label. name -> openml id,
# breast cancer comes with sklearn. openml downloads once and caches in ~/scikit_learn_data
REAL_DATASETS = {"breast cancer": None, "banknote": 1462, "spambase": 44, "magic": 1120, "pima": 37,
                 "ionosphere": 59, "phoneme": 1489, "qsar": 1494, "blood": 1464}

# a real dataset, split into train and test (stratified, 20 % test) by seed, standardized
# on the training points, and every entry hidden with probability missing_rate
# (hide_entries). returns the same six tensors as load_data
def load_real(name, missing_rate, seed, device):
    if REAL_DATASETS[name] is None:
        x, y = load_breast_cancer(return_X_y=True)
    else:
        data = fetch_openml(data_id=REAL_DATASETS[name], as_frame=False, parser="auto")
        x = np.asarray(data.data, dtype=float)
        y = (data.target == np.unique(data.target)[1]).astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(x.astype(np.float32), y.astype(np.int64), test_size=0.2,
                                                        random_state=seed, stratify=y)
    # a constant feature (ionosphere has one) is only centred
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    rng = np.random.default_rng(seed)
    x_missing = hide_entries(x_train, missing_rate, rng)
    x_test_missing = hide_entries(x_test, missing_rate, rng)
    arrays = (x_train, x_missing, y_train, x_test, x_test_missing, y_test)
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]
