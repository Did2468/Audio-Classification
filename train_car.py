import numpy as np
import pennylane as qml
from sklearn.metrics import classification_report, confusion_matrix
import time

# ── CONFIG ───────────────────────────────────
N_QUBITS    = 8
N_LAYERS    = 3
N_EPOCHS    = 50
LR          = 0.01
BATCH_SIZE  = 32
RANDOM_SEED = 42
VECTOR_DIR  = "./quantum_ready"
CLASS_NAMES = ["car_clean", "car_knocking"]
# ─────────────────────────────────────────────

np.random.seed(RANDOM_SEED)

# ── LOAD ─────────────────────────────────────
X_train_all = np.load(f"{VECTOR_DIR}/X_train.npy")
X_test_all  = np.load(f"{VECTOR_DIR}/X_test.npy")
y_train_all = np.load(f"{VECTOR_DIR}/y_train.npy")
y_test_all  = np.load(f"{VECTOR_DIR}/y_test.npy")

# ── FILTER TO CAR CLASSES ONLY ───────────────
# car_clean = 0, car_knocking = 1 in original labels
# we keep those two and relabel as 0 and 1
mask_train = (y_train_all == 0) | (y_train_all == 1)
mask_test  = (y_test_all  == 0) | (y_test_all  == 1)

X_train = X_train_all[mask_train]
y_train = y_train_all[mask_train]   # already 0 and 1, no change needed
X_test  = X_test_all[mask_test]
y_test  = y_test_all[mask_test]

print("=" * 52)
print("  Binary VQC — car_clean vs car_knocking")
print("=" * 52)
print(f"\n  Train  : {X_train.shape[0]} samples")
print(f"    car_clean    : {(y_train == 0).sum()}")
print(f"    car_knocking : {(y_train == 1).sum()}")
print(f"\n  Test   : {X_test.shape[0]} samples")
print(f"    car_clean    : {(y_test == 0).sum()}")
print(f"    car_knocking : {(y_test == 1).sum()}")
print(f"\n  Qubits : {N_QUBITS}")
print(f"  Layers : {N_LAYERS}")
print(f"  Params : {N_LAYERS * N_QUBITS * 3}")
print(f"  Epochs : {N_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")

# ── DEVICE ───────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)


# ─────────────────────────────────────────────
# CIRCUIT
# ─────────────────────────────────────────────

def angle_encoding(x):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)


def learnable_layers(params):
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.Rot(
                params[layer, qubit, 0],
                params[layer, qubit, 1],
                params[layer, qubit, 2],
                wires=qubit
            )
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])


@qml.qnode(dev)
def vqc(x, params):
    """
    Binary circuit — measure only Z0.
      Z0 positive -> car_clean    (class 0)
      Z0 negative -> car_knocking (class 1)
    """
    angle_encoding(x)
    learnable_layers(params)
    return qml.expval(qml.PauliZ(0))


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_one(x, params):
    z0 = vqc(x, params)
    return 0 if z0 >= 0 else 1


def predict_batch(X, params):
    return np.array([predict_one(x, params) for x in X])


def accuracy(X, y, params):
    return (predict_batch(X, params) == y).mean()


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

def loss_one(x, label, params):
    """
    Binary cross entropy.
    Z0 is in [-1, +1].
    Convert to probability in [0, 1] via (1 - z0) / 2:
      z0 = +1  ->  prob of class 1 = 0.0  (certain car_clean)
      z0 =  0  ->  prob of class 1 = 0.5  (uncertain)
      z0 = -1  ->  prob of class 1 = 1.0  (certain car_knocking)
    """
    z0   = vqc(x, params)
    prob_knocking = (1 - z0) / 2       # probability of car_knocking
    prob_clean    = 1 - prob_knocking  # probability of car_clean

    if label == 0:
        return -np.log(prob_clean    + 1e-9)
    else:
        return -np.log(prob_knocking + 1e-9)


def batch_loss(X_batch, y_batch, params):
    return np.mean([loss_one(x, int(y), params)
                    for x, y in zip(X_batch, y_batch)])


# ─────────────────────────────────────────────
# PARAMETER SHIFT GRADIENT
# ─────────────────────────────────────────────

def compute_gradient(X_batch, y_batch, params):
    grad  = np.zeros_like(params)
    shift = np.pi / 2
    for idx in np.ndindex(params.shape):
        p_plus       = params.copy(); p_plus[idx]  += shift
        p_minus      = params.copy(); p_minus[idx] -= shift
        grad[idx]    = (batch_loss(X_batch, y_batch, p_plus) -
                        batch_loss(X_batch, y_batch, p_minus)) / 2
    return grad


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(params):
    n         = len(X_train)
    log_lines = []

    print(f"\n  {'Epoch':<8} {'Loss':>10} {'Train Acc':>12} {'Test Acc':>10} {'Time/epoch':>12}")
    print(f"  {'-' * 56}")

    for epoch in range(1, N_EPOCHS + 1):
        t_start  = time.time()
        idx      = np.random.permutation(n)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, BATCH_SIZE):
            X_batch    = X_s[start : start + BATCH_SIZE]
            y_batch    = y_s[start : start + BATCH_SIZE]
            grad       = compute_gradient(X_batch, y_batch, params)
            params    -= LR * grad
            epoch_loss += batch_loss(X_batch, y_batch, params)
            n_batches  += 1

        epoch_loss /= n_batches
        elapsed     = time.time() - t_start

        train_acc = accuracy(X_train, y_train, params)
        test_acc  = accuracy(X_test,  y_test,  params)
        line = (f"  {epoch:<8} {epoch_loss:>10.4f} "
                f"{train_acc:>11.2%} {test_acc:>10.2%} {elapsed:>11.1f}s")
        print(line)
        log_lines.append(line)

        if epoch % 10 == 0:
            np.save(f"binary_params_epoch{epoch}.npy", params)
            print(f"  checkpoint saved -> binary_params_epoch{epoch}.npy")

    return params, log_lines


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(params):
    print(f"\n{'=' * 52}")
    print(f"  Final Evaluation on Test Set")
    print(f"{'=' * 52}")

    preds    = predict_batch(X_test, params)
    test_acc = (preds == y_test).mean()

    print(f"\n  Test Accuracy : {test_acc:.2%}\n")
    print(classification_report(
        y_test, preds,
        target_names=CLASS_NAMES,
        digits=3
    ))

    print("  Confusion Matrix:")
    print(f"  {'':20} {'car_clean':>12} {'car_knocking':>14}")
    cm = confusion_matrix(y_test, preds)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:20}" +
              "  ".join([f"{v:>12}" for v in row]))

    return test_acc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    params = np.random.uniform(0, 2 * np.pi,
                               size=(N_LAYERS, N_QUBITS, 3))

    print(f"\n  Baseline (random params):")
    baseline = accuracy(X_test, y_test, params)
    print(f"  Test accuracy = {baseline:.2%}  (random chance = 50.00%)")

    print(f"\n  Starting training...")
    params, log_lines = train(params)

    np.save("trained_params_binary.npy", params)
    with open("training_log_binary.txt", "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Saved -> trained_params_binary.npy")
    print(f"  Saved -> training_log_binary.txt")

    evaluate(params)
