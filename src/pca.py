import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


vector_dir = "./vectors"
output_dir = "./quantum_ready"

N_COMPONENTS = 8
TEST_SIZE = 0.2
RANDOM_SEED = 67


os.makedirs(output_dir,exist_ok = True)

# Loading the data 
print("="*52)
print("Starting the PCA and Normalization Part")
print("="*52)

X = np.load(f"{vector_dir}/features.npy")
y = np.load(f"{vector_dir}/labels.npy")

label_names = np.load(f"{vector_dir}/labels_names.npy")

print("Loaded")
print(f"X:{X.shape}")
print(f"y:{y.shape}")
print(f"Classes:{list(label_names)}")

# training testing and splitting

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = TEST_SIZE,stratify=y,random_state=RANDOM_SEED)

print(f"Train Test split 80/20 Successfull")
print(f"Train:{X_train.shape}")
print(f"Test:{X_test.shape}")
print(f"Train class counts:")

for i,name in enumerate(label_names):
	print(f"{i}{name}: {(y_train==i).sum()}")

# PCA PART(fitting only the training data)

pca = PCA(n_components=N_COMPONENTS,random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"Succesfully transformed 34 Dimensions into {N_COMPONENTS} Dimensions")
print(f"Variance captured per component:")

for i,(e,c) in enumerate(zip(explained,cumulative)):
	bar = "*"* int(e*40)
	print(f"PC{i+1:02d}: {e:.2} (cumulative{c:.2%}  {bar}")

print(f" Total variance retained: {cumulative[-1]:.2%}")


# NORMALIZATION part (zero to pi)
scaler = MinMaxScaler(feature_range=(0,np.pi))
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.fit_transform(X_test_pca)

# Saving 
np.save(f"{output_dir}/X_train.npy",  X_train_scaled)
np.save(f"{output_dir}/X_test.npy",  X_test_scaled)
np.save(f"{output_dir}/y_train.npy", y_train)
np.save(f"{output_dir}/y_test.npy", y_test)
np.save(f"{output_dir}/label_names.npy",label_names)

joblib.dump(pca,f"{output_dir}/pca.pk1")
joblib.dump(scaler, f"{output_dir}/scaler.pk1")

print(f"Saved to{output_dir}")
