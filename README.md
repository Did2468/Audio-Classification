## The Data

The dataset contains .wav recordings of car and truck engines under two conditions —
healthy (clean) and faulty (knocking). There are approximately 300 files per class,
1199 files in total, making it a balanced 4-class classification problem.

Classes:
  - car_clean      — healthy car engine
  - car_knocking   — car engine with a knocking fault
  - truck_clean    — healthy truck engine
  - truck_knocking — truck engine with a knocking fault

The knocking sound is a percussive, low-frequency fault pattern caused by improper
combustion timing. It is acoustically distinct enough that even raw spectral features
capture it clearly, as confirmed during the PCA step.

Raw .wav files are stored under data/ and are not tracked by git due to size.

---

## The Pipeline

The core idea is to convert audio files into fixed-length numerical vectors,
compress and normalize them to fit into qubits, encode them into a quantum circuit,
and train a variational classifier on top.

Step 1 — Audio to Feature Vectors

Each .wav file is loaded with librosa and converted into a 34-dimensional vector
by extracting the following features:

  - MFCCs (13 coefficients, mean + std over time) — 26 dims
    Captures timbral texture. The knocking fault introduces harmonic distortion
    that shows up clearly in the mid-range MFCC coefficients.

  - Spectral centroid (mean + std) — 2 dims
    The frequency center of mass. Trucks sit lower than cars here.

  - Spectral rolloff (mean + std) — 2 dims
    The frequency below which 85% of the energy lies.

  - RMS energy (mean + std) — 2 dims
    Overall loudness. Knock events produce sharp intermittent spikes.

  - Zero-crossing rate (mean + std) — 2 dims
    How often the signal crosses zero. Higher for transient/noisy content.

One audio file becomes one row of 34 numbers. The full dataset becomes a
matrix of shape (1199, 34). Code: src/audio_to_vectors.py
Output saved to: vectors/features.npy, vectors/labels.npy


Step 2 — PCA and Normalization

Two things need to happen before feeding vectors into a quantum circuit.

First, dimensionality reduction. A quantum simulator tracks 2^n amplitudes in
memory where n is the number of qubits. 34 qubits would require tracking 2^34
 amplitudes which is not feasible. PCA compresses 34 dimensions
down to 8 while retaining 99.96% of the variance. The compression is this clean
because the 34 features are highly correlated — they all respond to the same
underlying physical events (knock or no knock), so most of the information
collapses into very few directions. PC01 alone captures 88.5% of all variance.

Second, normalization. Quantum encoding maps each number to a rotation angle on
a qubit. This only makes sense in the range [0, pi]. A MinMaxScaler scales each
of the 8 PCA components to this range.

The PCA and scaler are fit only on training data and then applied to test data
to avoid data leakage. Both fitted objects are saved to disk for use during
inference on new audio files.

Code: src/pca.py
Output saved to: quantum_ready/X_train.npy, X_test.npy, y_train.npy, y_test.npy,
                 quantum_ready/pca.pkl, quantum_ready/scaler.pkl
The pkl files are packages created by joblib to be instantly able to load trained models 
Train/test split: 80/20 stratified
  Train: 959 samples (240/239/240/240 per class)
  Test:  240 samples


Step 3 — Qubit Encoding

The 8-dimensional normalized vectors are encoded into 8-qubit quantum states
using a ZZFeatureMap. This is the bridge between classical and quantum.

The encoding works in two layers:

  Layer 1 — individual feature encoding
    A Hadamard gate puts each qubit into superposition.
    Then Rz(x[i]) encodes feature i as a phase rotation on qubit i.

  Layer 2 — pairwise feature interactions
    For each adjacent pair of qubits (i, i+1):
      CNOT -> Rz((pi - x[i]) * (pi - x[i+1])) -> CNOT
    This cross term encodes the product of two features into the quantum state,
    allowing the circuit to detect correlated patterns like high RMS and high ZCR
    occurring together, which is characteristic of a knocking engine.

The encoding was verified by checking state overlaps between classes.
Overlap close to 0 means the quantum states are distinct (good).
Overlap close to 1 means they are collapsed together (bad).

Results:
  car_clean vs car_knocking    — 0.019  (very distinct)
  car_knocking vs truck_clean  — 0.011  (very distinct)
  truck_clean vs truck_knocking — 0.150  (distinct)
  car_clean vs truck_clean     — 0.386  (most similar, expected since both are healthy)

The encoding preserves the real-world relationships between classes. Classes that
sound more alike in the physical world are also closer in quantum state space.

Note: the vectors in quantum_ready/ are plain numpy arrays. They become quantum
states only when passed into the circuit at runtime. Nothing is saved after encoding —
the quantum state exists only inside the simulator's memory during a forward pass.

Code: src/encoding.py
Mainly this id done to check whether the data vectors we have are valid for our classifier the probabilities summed upto one 
we can se that the pauli z expressions all became 0 but not to worry it is expected later when w e ansatz it it will be restored and have normal value

Step 4 — Ansatz (in progress)

Step 5 — Training and Evaluation (pending)

---

## Current Status

  Step 1  done — feature extraction verified, 1199 vectors of shape (34,)
  Step 2  done — PCA and normalization verified, 99.96% variance retained
  Step 3  done — encoding verified, classes well separated in quantum state space
  Step 4  in progress
  Step 5  pending

---

## Future Plan

- Complete the ansatz and verify the full circuit structure (Step 4)
- Implement the training loop using parameter-shift gradient descent (Step 5)
- Evaluate on the test set and record classification accuracy per class
- Update this README with results once training is complete

---

## Repository Structure

  src/audio_to_vectors.py   — audio feature extraction
  src/pca.py      — PCA and normalization
  src/encoding.py           — qubit encoding verification

  data/                       — raw .wav files
  vectors/                    — intermediate feature vectors
  quantum_ready/              — circuit-ready normalized vectors

---

## Setup

  pip install scikit-learn matplotlib librosa numpy  

  python src/audio_to_vectors.py
  python src/pca.py
  python src/encoding.py

---

## Stack

  librosa       — audio feature extraction
  scikit-learn  — PCA, normalization, train/test split, evaluation
  PennyLane     — quantum circuit simulation
  numpy         — numerical operations

---

Author: Likhith Reddy Kaliki
