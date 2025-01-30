import numpy as np

data = np.load(r"D:\MAPUA\THESIS2_VER3\scripts\VID_20240128_150039_vectors.npy")
print(f"Shape of the data: {data.shape}")
print(f"Data type of the array: {data.dtype}")

num_frames = data.shape[0]  # Get the number of frames

for i in range(num_frames):
    print(f"--- Frame {i} ---")
    print(f"  Shape: {data[i].shape}")
    print(f" Full Frame Data: \n {data[i]}")