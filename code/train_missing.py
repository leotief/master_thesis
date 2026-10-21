import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from data import DataManager
from gmm import GMM
from pcpg import PCPGMCHermite, PCPGMCHermiteDiff

# ── config ────────────────────────────────────────────────────────────────────

seed         = 42
missing_rate = 0.4
n_features   = 10
K            = 4
n_gauss_quad = 20
n_pg_eval    = 200   # PG samples at evaluation time
n_pg_train   = 30    # PG samples during classifier training (fewer = faster)
gmm_epochs   = 80
clf_epochs   = 30

np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── data ──────────────────────────────────────────────────────────────────────

dm = DataManager("sk_class", samples=2000, features=n_features,
                 test_size=0.2, random_state=seed)
x_train, x_test, y_train, y_test = dm.get_data()

missing_loader         = dm.get_missing_loader(missing_rate=missing_rate, batch_size=32, seed=seed)
x_test_miss, y_test_t = dm.get_test_missing_batch(missing_rate=missing_rate, seed=seed, device=device)

print(f"Dataset: {len(x_train)} train / {len(x_test)} test | "
      f"{n_features} features | {missing_rate*100:.0f}% missing\n")

# ── training utilities ────────────────────────────────────────────────────────

def fit_classifier_pcpg(w_init, estimator, loader, epochs=20, lr=0.01, n_pg=50):
    w         = nn.Parameter(w_init.clone().to(estimator.model.device))
    optimizer = torch.optim.Adam([w], lr=lr)

    print("Fitting classifier (PCPG)...", end=" ", flush=True)
    t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(estimator.model.device)
            y_batch = y_batch.to(estimator.model.device).float()

            # Each row may have a different missingness pattern — loop required.
            preds = torch.stack([
                estimator.forward(w, x_batch[i], n_pg=n_pg)
                for i in range(len(x_batch))
            ])

            preds = preds.clamp(1e-7, 1 - 1e-7)
            loss  = -(y_batch * preds.log() + (1 - y_batch) * (1 - preds).log()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"\n  epoch {epoch+1}/{epochs}  loss={epoch_loss/n_batches:.4f}", end=" ", flush=True)

    print(f"\ndone ({time.perf_counter() - t0:.1f}s)")
    return w.detach()

def marginal_predictions(w, x_missing_batch, estimator, n_pg):
    preds = np.array([
        estimator(w, x_missing_batch[i], n_pg=n_pg)
        for i in range(len(x_missing_batch))
    ])
    return np.clip(preds, 1e-7, 1 - 1e-7)

# ── step 1: GMM on missing data ───────────────────────────────────────────────

means_init = torch.tensor(x_train[:K], dtype=torch.float32, device=device)
px = GMM(device, K=K, means_init=means_init, n_features=n_features).to(device)
px.fit(missing_loader, epochs=gmm_epochs)

with torch.no_grad():
    mll = px(x_test_miss[:50].to(device)).mean().item()
print(f"GMM marginal log-lik on missing test (first 50): {mll:.3f}\n")

# ── step 2: w_full — oracle on complete data ──────────────────────────────────

print("Training w_full (sklearn LR on complete data)...")
clf_full = LogisticRegression(max_iter=1000, random_state=seed)
clf_full.fit(x_train, y_train)
w_full = torch.tensor(clf_full.coef_[0], dtype=torch.float32, device=device)
print(f"  complete-data accuracy: {clf_full.score(x_test, y_test):.3f}\n")

# ── step 3: w_impute — GMM single-imputation then sklearn LR ─────────────────

print("Training w_impute (sklearn LR on GMM-imputed missing data)...")
x_miss_t = next(iter(
    dm.get_missing_loader(missing_rate=missing_rate, batch_size=len(x_train), seed=seed)
))[0].to(device)

with torch.no_grad():
    imputed_rows = [px.sample(1, x_obs=x_miss_t[i])[0].cpu().numpy()
                    for i in range(len(x_miss_t))]
x_imputed_np = np.stack(imputed_rows)

clf_impute = LogisticRegression(max_iter=1000, random_state=seed)
clf_impute.fit(x_imputed_np, y_train)
w_impute = torch.tensor(clf_impute.coef_[0], dtype=torch.float32, device=device)
print(f"  imputed-data accuracy (complete test): {clf_impute.score(x_test, y_test):.3f}\n")

# ── step 4: w_pcpg — PCPG marginal-prediction training ───────────────────────

estimator_train = PCPGMCHermiteDiff(px, n_gauss_quad=n_gauss_quad, seed=seed)
w_pcpg = fit_classifier_pcpg(
    w_init    = torch.zeros(n_features, device=device),
    estimator = estimator_train,
    loader    = missing_loader,
    epochs    = clf_epochs,
    lr        = 0.05,
    n_pg      = n_pg_train,
)
print()

# ── step 5: evaluate all three on the missing test set ───────────────────────

print("Evaluating on missing test set using PCPGMCHermite...\n")
estimator_eval = PCPGMCHermite(px, n_gauss_quad=n_gauss_quad, seed=seed)
y_true         = y_test_t.cpu().numpy()

results = {}
for name, w in [("w_full (oracle)", w_full), ("w_impute", w_impute), ("w_pcpg", w_pcpg)]:
    t0    = time.perf_counter()
    preds = marginal_predictions(w, x_test_miss, estimator_eval, n_pg=n_pg_eval)
    elapsed = time.perf_counter() - t0

    acc     = ((preds > 0.5).astype(int) == y_true).mean()
    logloss = -(y_true * np.log(preds + 1e-7) + (1 - y_true) * np.log(1 - preds + 1e-7)).mean()
    results[name] = preds
    print(f"  {name:<22}  acc={acc:.3f}  log-loss={logloss:.4f}  ({elapsed:.1f}s)")

# ── step 6: per-point sample table ───────────────────────────────────────────

print("\nSample predictions (first 10 test points):")
print(f"  {'idx':>3}  {'y':>3}  {'w_full':>8}  {'w_impute':>8}  {'w_pcpg':>8}")
for i in range(10):
    print(f"  {i:>3}  {int(y_true[i]):>3}  "
          f"{results['w_full (oracle)'][i]:>8.3f}  "
          f"{results['w_impute'][i]:>8.3f}  "
          f"{results['w_pcpg'][i]:>8.3f}")

print("\nAll predictions are E[σ(wᵀx)|x_obs] — marginal probability under missing features.")
