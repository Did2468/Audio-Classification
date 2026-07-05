import numpy as np
import pennylane as qml
from sklearn.metrics import classification_report, confusion_matrix
import time

# ── CONFIG ───────────────────────────────────
N_QUBITS    = 8
N_LAYERS    = 3
N_EPOCHS    = 25
LR          = 0.05
BATCH_SIZE  = 16                        #changed batch size and learning rate
RANDOM_SEED = 42
VECTOR_DIR  = "./quantum_ready"
# ─────────────────────────────────────────────

np.random.seed(RANDOM_SEED)

# ── LOAD ─────────────────────────────────────
X_train     = np.load(f"{VECTOR_DIR}/X_train.npy")
X_test      = np.load(f"{VECTOR_DIR}/X_test.npy")
y_train     = np.load(f"{VECTOR_DIR}/y_train.npy")
y_test      = np.load(f"{VECTOR_DIR}/y_test.npy")
label_names = np.load(f"{VECTOR_DIR}/label_names.npy")

print("=" * 52)
print("  VQC Training — Manual Implementation")
print("=" * 52)
print(f"\n  Train  : {X_train.shape[0]} samples")
print(f"  Test   : {X_test.shape[0]} samples")
print(f"  Qubits : {N_QUBITS}")
print(f"  Layers : {N_LAYERS}")
print(f"  Params : {N_LAYERS * N_QUBITS * 3}")
print(f"  Epochs : {N_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")

# ── DEVICE ───────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)


# ─────────────────────────────────────────────
# CIRCUIT
# ─────────────────────────────────────────────

def angle_encoding(x):
    """
    Load 8-dim vector into 8 qubits.
    Ry(x[i]) rotates qubit i by angle x[i].
    Fixed — no learnable parameters here.
    """
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)


def learnable_layers(params):
    """
    3 learnable layers. Each layer:
      Step 1 — Rot(phi, theta, omega) on every qubit
               3 learnable angles per qubit, can point anywhere on Bloch sphere
      Step 2 — ring CNOT entanglement
               q0->q1->q2->...->q7->q0
               mixes information across all qubits

    params shape: (N_LAYERS, N_QUBITS, 3)
    """
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
    Full circuit:
        angle encoding  ->  learnable layers  ->  measure

    Returns PauliZ expectation on q0 and q1.
    Sign combination maps to 4 classes:
        (+, +) -> car_clean      (class 0)
        (+, -) -> car_knocking   (class 1)
        (-, +) -> truck_clean    (class 2)
        (-, -) -> truck_knocking (class 3)
    """
    angle_encoding(x)
    learnable_layers(params)
    return [qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1))]


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_one(x, params):
    z0, z1 = vqc(x, params)
    b0 = 0 if z0 >= 0 else 1
    b1 = 0 if z1 >= 0 else 1
    return b0 * 2 + b1


def predict_batch(X, params):
    return np.array([predict_one(x, params) for x in X])


def accuracy(X, y, params):
    preds = predict_batch(X, params)
    return (preds == y).mean()


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def loss_one(x, label, params):
    z0, z1 = vqc(x, params)

    # convert Z measurements to probabilities in [0,1]
    p0 = (1 - z0) / 2    # prob that qubit 0 is in south (|1>)
    p1 = (1 - z1) / 2    # prob that qubit 1 is in south (|1>)

    # true binary targets for each qubit based on class
    # class 0 car_clean      -> b0=0, b1=0  (both north)
    # class 1 car_knocking   -> b0=0, b1=1  (north, south)
    # class 2 truck_clean    -> b0=1, b1=0  (south, north)
    # class 3 truck_knocking -> b0=1, b1=1  (both south)
    b0 = 1 if label >= 2 else 0
    b1 = 1 if label % 2 == 1 else 0

    # binary cross entropy for each qubit independently
    loss_q0 = -(b0 * np.log(p0 + 1e-9) + (1-b0) * np.log(1-p0 + 1e-9))
    loss_q1 = -(b1 * np.log(p1 + 1e-9) + (1-b1) * np.log(1-p1 + 1e-9))

    return loss_q0 + loss_q1


def batch_loss(X_batch, y_batch, params):
    return np.mean([loss_one(x, int(y), params)
                    for x, y in zip(X_batch, y_batch)])


# ─────────────────────────────────────────────
# PARAMETER SHIFT GRADIENT
# ─────────────────────────────────────────────

def compute_gradient(X_batch, y_batch, params):
    """
    Parameter shift rule.

    For every parameter theta_i:
        gradient_i = [ L(theta_i + pi/2) - L(theta_i - pi/2) ] / 2

    This is the exact gradient of the quantum circuit.
    Requires 2 circuit evaluations per parameter per batch.
    Total evaluations per batch = 2 x 72 = 144.
    """
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
    print(f"  {'-'*56}")

    for epoch in range(1, N_EPOCHS + 1):
        t_start    = time.time()
        idx        = np.random.permutation(n)
        X_s, y_s   = X_train[idx], y_train[idx]

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

        # save checkpoint every 10 epochs
        if epoch % 10 == 0:
            np.save(f"params_epoch{epoch}.npy", params)
            print(f"  checkpoint saved -> params_epoch{epoch}.npy")

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
        target_names=label_names,
        digits=3
    ))

    print("  Confusion Matrix:")
    print("  " + " " * 20 + "  ".join([f"{n[:10]:>10}" for n in label_names]))
    cm = confusion_matrix(y_test, preds)
    for i, row in enumerate(cm):
        print(f"  {label_names[i][:20]:20}" +
              "  ".join([f"{v:>10}" for v in row]))

    return test_acc, preds


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    params = np.random.uniform(0, 2 * np.pi,
                               size=(N_LAYERS, N_QUBITS, 3))

    print(f"\n  Baseline (random params, no training):")
    baseline = accuracy(X_test, y_test, params)
    print(f"  Test accuracy = {baseline:.2%}  (random chance = 25.00%)")

    print(f"\n  Starting training...")
    params, log_lines = train(params)

    np.save("trained_params.npy", params)
    with open("training_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Saved -> trained_params.npy")
    print(f"  Saved -> training_log.txt")

    evaluate(params)
