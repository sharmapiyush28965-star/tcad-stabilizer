"""
Phase 4: M1 — global MLP that predicts damping alpha from state features.

Architecture: 6 -> 32 -> 16 -> 1, tanh hidden, sigmoid output (alpha in (0,1]).
Trained with full-batch Adam on MSE. Pure numpy (no dependencies).
"""
import numpy as np
import json


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class M1:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/6), (6, 32))
        self.b1 = np.zeros(32)
        self.W2 = rng.normal(0, np.sqrt(2/32), (32, 16))
        self.b2 = np.zeros(16)
        self.W3 = rng.normal(0, np.sqrt(2/16), (16, 1))
        self.b3 = np.zeros(1)
        # feature standardization (fit on train)
        self.mu = np.zeros(6)
        self.sd = np.ones(6)

    def forward(self, X):
        Xn = (X - self.mu) / self.sd
        self.z1 = Xn @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def predict(self, X):
        return self.forward(np.atleast_2d(X)).ravel()

    def fit(self, X, y, Xv, yv, epochs=2000, lr=3e-3, verbose=True):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        n = len(X)
        # Adam state
        params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        y = y.reshape(-1, 1); yv = yv.reshape(-1, 1)
        best_vl, best = float("inf"), None
        for ep in range(1, epochs + 1):
            p = self.forward(X)  # (n,1)
            # backprop (MSE)
            d3 = 2 * (p - y) / n * p * (1 - p)
            dW3 = self.a2.T @ d3; db3 = d3.sum(0)
            d2 = (d3 @ self.W3.T) * (1 - self.a2 ** 2)
            dW2 = self.a1.T @ d2; db2 = d2.sum(0)
            d1 = (d2 @ self.W2.T) * (1 - self.a1 ** 2)
            Xn = (X - self.mu) / self.sd
            dW1 = Xn.T @ d1; db1 = d1.sum(0)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                     "W3": dW3, "b3": db3}
            for p_ in params:
                m[p_] = b1 * m[p_] + (1 - b1) * grads[p_]
                v[p_] = b2 * v[p_] + (1 - b2) * grads[p_] ** 2
                mh = m[p_] / (1 - b1 ** ep); vh = v[p_] / (1 - b2 ** ep)
                setattr(self, p_, getattr(self, p_) -
                        lr * mh / (np.sqrt(vh) + eps))
            if ep % 200 == 0 or ep == epochs:
                vl = float(((self.predict(Xv) - yv.ravel()) ** 2).mean())
                tl = float(((p.ravel() - y.ravel()) ** 2).mean())
                if verbose:
                    print(f"  ep {ep}: train_mse={tl:.4f} val_mse={vl:.4f}",
                          flush=True)
                if vl < best_vl:
                    best_vl = vl
                    best = {p_: getattr(self, p_).copy() for p_ in params}
        for p_ in params:
            setattr(self, p_, best[p_])
        return best_vl

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path):
        m = cls()
        z = np.load(path)
        for k in ("W1", "b1", "W2", "b2", "W3", "b3", "mu", "sd"):
            setattr(m, k, z[k])
        return m


FEAT = ["iteration", "log_honest", "max_delta", "mean_delta",
        "max_dpsi_vt", "mean_dpsi_vt"]


def load_split(path="dataset_v1_split.json"):
    d = json.load(open(path))
    def mat(split):
        X = np.array([[s["features"][k] for k in FEAT] for s in d[split]])
        y = np.array([s["alpha"] for s in d[split]])
        return X, y
    return mat("train"), mat("val"), mat("test")


if __name__ == "__main__":
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_split()
    print(f"train={len(Xtr)} val={len(Xv)} test={len(Xte)}")
    m1 = M1(seed=0)
    m1.fit(Xtr, ytr, Xv, yv, epochs=2000, lr=3e-3)
    for name, (X, y) in (("train", (Xtr, ytr)), ("val", (Xv, yv)),
                               ("test", (Xte, yte))):
        p = m1.predict(X)
        ss = ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mae = np.abs(y - p).mean()
        print(f"{name}: R^2={1-ss:.3f} MAE={mae:.4f}")
    m1.save("m1_weights.npz")
    print("saved m1_weights.npz")
