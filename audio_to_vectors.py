import librosa
import numpy as np
import os
from pathlib import Path

data_root = "./data"

output_root = "./vectors"

SR = 22050

N_MFCC = 13

os.makedirs(output_root,exist_ok = True)

def extract_features(filepath:str)->np.ndarray:
	y,_ = librosa.load(filepath,sr = SR,mono = True)
	
	mfcc = librosa.feature.mfcc(y=y,sr=SR,n_mfcc=N_MFCC)
	mfcc_feat = np.concatenate([mfcc.mean(axis=1),mfcc.std(axis=1)])
	
	centroid = librosa.feature.spectral_centroid(y=y,sr=SR)[0]
	centroid_feat = np.array([centroid.mean(),centroid.std()])
	
	rolloff = librosa.feature.spectral_rolloff(y=y,sr=SR)[0]
	rolloff_feat = np.array([rolloff.mean(),rolloff.std()])

	rms = np.sqrt(np.mean(y**2))
	rms_std = np.sqrt(np.var(y))
	rms_feat = np.array([rms,rms_std])

	zcr = librosa.feature.zero_crossing_rate(y)[0]
	zcr_feat = np.array([zcr.mean(),zcr.std()])

	return np.concatenate([mfcc_feat,centroid_feat,rolloff_feat,rms_feat,zcr_feat])


def process_dataset(data_root:str):
	classes = ["car_clean","car_knocking","truck_clean","truck_knocking"]
	X, y = [],[]
	skipped = 0
	
	for label,cls in enumerate(classes):
		folder = Path(data_root) / cls
		if not folder.exists():
			print("Folder not found!!!!")
			continue
		files = sorted(folder.glob("*.wav"))
		print(f"Found {len(files)} files")
		
		for f in files:
			try:
				vec = extract_features(str(f))
				X.append(vec)
				y.append(label)
				print(f"{f.name} -> shape{vec.shape}")
			except Exception as e:
				print(f"Error found with file cant extract features{type(e).__name__}: {e}")
				skipped +=1
	return np.array(X),np.array(y),classes,skipped


if __name__ == "__main__":
	print("+" * 52)
	print("Audio to vector converter(feature extraction)")
	print("+" * 52)
	X,y ,class_names,skipped = process_dataset(data_root)
	
	if(len(X)==0):
		print("Invalid data root path or no files found at the path")
	else:
		np.save(f"{output_root}/features.npy",  X)
		np.save(f"{output_root}/labels.npy",   y)
		np.save(f"{output_root}/labels_names.npy",  class_names)
		
		print("="*52)
		print(f"Done processing the data")
		print(f"Skipped files: {skipped}")
		print(f"X shape: {X.shape}")
		print(f"y shape: {y.shape}")
		print(f"Saved to {output_root}")
		print(f"features.npy - {X.nbytes/1024:.1f} KB")
		print(f"labels.npy")
		print(f"label_names.npy")
		print(f"Per-class counts:")
		for i,name in enumerate(class_names):
			print(f"{i} {name}: {(y==i).sum()} samples")
	
		print(f" Sample vector(first file):")
		print(f" {X[0].round(4)}")
