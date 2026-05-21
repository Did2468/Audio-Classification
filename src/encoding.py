import pennylane as qml
import numpy as np

N_QUBITS = 8
vector_dir = "./quantum_ready"


X_train = np.load(f"{vector_dir}/X_train.npy")
y_train = np.load(f"{vector_dir}/y_train.npy")
label_names = np.load(f"{vector_dir}/label_names.npy")


dev = qml.device("default.qubit",wires = N_QUBITS)


def zz_feature_map(x):
	#Layer 1
	for i in range(N_QUBITS):
		qml.Hadamard(wires=i)
		qml.RZ(x[i],wires=i)
	#Layer 2
	for i in range(N_QUBITS - 1):
        	qml.CNOT(wires=[i, i + 1])
        	qml.RZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        	qml.CNOT(wires=[i, i + 1])
	
#Defining the Qnodes

@qml.qnode(dev)

def encode_return(x):
	#encode x into qubits and rreutn the full statevrecor

	zz_feature_map(x)
	
	return qml.state()

@qml.qnode(dev)

def encode_measure(x):
	#enode x and return pauliZ expectation on all Qubits

	zz_feature_map(x)
	return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


if __name__=="__main__":
	print("="*52)
	print("Verfication of qubit encoding")
	print("="*52)
	
	print("Input vectors(one per class):")
	samples = {}
	for cls_idx,name in enumerate(label_names):
		idx = np.where(y_train==cls_idx)[0][0]
		samples[name] = X_train[idx]
		print(f"{name}:")
		print(f"{X_train[idx].round(4)}")
	# Statevector
	print("\n[2] Statevector after ZZFeatureMap encoding:")
	print("(256 complex amplitudes *  showing first 8)")
	x0 = list(samples.values())[0]
	state = encode_return(x0)
	print(f"\n    Full state dimension : {len(state)}")
	print(f"    First 8 amplitudes   :")
	for i, amp in enumerate(state[:8]):
        	prob = abs(amp) ** 2
        	print(f"      |{i:08b}* : {amp.real:+.4f} + {amp.imag:+.4f}i   "
              	f"prob={prob:.4f}")
	print(f"\n    Sum of all probabilities : "f"{sum(abs(a)**2 for a in state):.6f}  (must be 1.0)")

	#pauli z measurements
	print("\n[3] PauliZ expectation values per qubit (one per class):")
	print("    Values in [-1, +1]. Different classes should give different patterns.\n")
	print(f"    {'Class':<20} " + "  ".join([f"Q{i}" for i in range(N_QUBITS)]))
	print("    " + "-" * 50)
	for name, x in samples.items():
		z_vals = encode_measure(x)
		vals   = "  ".join([f"{v:+.2f}" for v in z_vals])
		print(f"    {name:<20} {vals}")

	#distinguishability
	print("\n[4] State distinguishability check:")
	print("    Inner product between encoded states of different classes.")
	print("    Close to 1.0 = states are similar (bad)")
	print("    Close to 0.0 = states are distinct  (good)\n")
	
	states = {}
	for name, x in samples.items():
		states[name] = encode_return(x)
	
	class_list = list(samples.keys())
	for i in range(len(class_list)):
		for j in range(i + 1, len(class_list)):
			n1, n2 = class_list[i], class_list[j]
			overlap = abs(np.dot(states[n1].conj(), states[n2])) ** 2
			bar = "*" * int(overlap * 20)
			print(f"    {n1} vs {n2}")
			print(f"      overlap = {overlap:.4f}  {bar}")
	#printing the circuit
	print("\n[5] Circuit diagram (ZZFeatureMap on first sample):")
	print()
	x0 = list(samples.values())[0]
	print(qml.draw(encode_measure)(x0))
	
	print(f"\n{'=' * 52}")
	print(f"  Encoding verified. Ready for ansatz + training.")
	print(f"{'=' * 52}")
