# toy datasets and helpers to make copies with missing values
#
# the data itself is always complete, missingness is added afterwards by setting
# entries to nan. everything else in the code uses the same convention: a nan
# means missing, and obs_mask = ~torch.isnan(x) gives the observed features.
# features are dropped completely at random (MCAR), independent of their value.

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class DataManager:
    # generates the dataset and splits it into train and test
    # 'sk_class' is sklearn's make_classification (gaussian clusters, any dimension,
    # balanced classes), 'moons' the 2d moons, mostly for plotting
    def __init__(self, name, samples=1000, features=20, test_size=0.2, random_state=42):
        if name == 'sk_class':
            x, y = make_classification(
                n_samples=samples,
                n_features=features,
                weights=[0.5, 0.5],
                random_state=random_state,
            )
        elif name == 'moons':
            x, y = make_moons(n_samples=samples, noise=0.1, random_state=random_state)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        # float32 for torch, labels as int64
        x = x.astype(np.float32)
        y = y.astype(np.int64)
        # stratified so both splits have the same class balance
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    

    # loaders on the complete data, used to fit the gmm in demo.py
    def get_dataloaders(self, batch_size=256):
        train_dataset = TensorDataset(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.y_test))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return train_loader, test_loader
    
    # raw numpy arrays, e.g. for sklearn's LogisticRegression
    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    # one complete test point. used to see what the prediction would be if nothing
    # was missing (not the target of the estimators, that is the expectation)
    def get_test(self, idx=0, device=None):
        x = torch.tensor(self.x_test[idx].copy())
        return x.to(device) if device is not None else x

    # same test point with exactly int(n_features * missing_rate) features set to nan.
    # this is the x_partial that all estimators take as input
    def get_test_missing(self, missing_rate=0.1, idx=0, device=None):
        x = self.x_test[idx].copy()
        n_missing = int(len(x) * missing_rate)
        # which features are dropped is random, using the global numpy rng,
        # so it depends on the seed set by the calling script
        missing_idx = np.random.choice(len(x), n_missing, replace=False)
        x[missing_idx] = np.nan
        x = torch.tensor(x)
        return x.to(device) if device is not None else x

    def get_missing_loader(self, missing_rate=0.3, batch_size=64, seed=42):
        """Training DataLoader with features independently set to NaN at rate missing_rate."""
        # used in train_missing.py, where also the training data is incomplete.
        # each entry is dropped on its own, so rows have different patterns
        # and the number missing per row varies
        rng    = np.random.default_rng(seed)
        x_miss = self.x_train.astype(np.float32).copy()
        x_miss[rng.random(x_miss.shape) < missing_rate] = np.nan
        dataset = TensorDataset(
            torch.from_numpy(x_miss),
            torch.from_numpy(self.y_train),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_test_missing_batch(self, missing_rate=0.3, seed=42, device=None):
        """Returns (x_test, y_test) tensors with features independently set to NaN."""
        # same as above but for the whole test set at once, for evaluation
        rng   = np.random.default_rng(seed)
        x_miss = self.x_test.astype(np.float32).copy()
        x_miss[rng.random(x_miss.shape) < missing_rate] = np.nan
        x_t = torch.from_numpy(x_miss)
        y_t = torch.from_numpy(self.y_test)
        if device is not None:
            return x_t.to(device), y_t.to(device)
        return x_t, y_t


 