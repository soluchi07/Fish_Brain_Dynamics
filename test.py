import h5py
import numpy as np

def diagnose_h5_ordering(h5_path, dataset_name='Data', num_planes=7):
    with h5py.File(h5_path, 'r') as f:
        # Load just the first 10 frames to keep memory usage minimal
        frames = f[dataset_name][:700].astype(np.float32)
        # --- 0 1 2 3 4 5 6 - 0 1 2 3 4 5 6 - 0 1 2 3 4 5 6 
        # --- 0 0 0 0 0 0 0 ... 1 1 1 1 1 1 ...

    # Flatten frames for correlation math
    f0 = frames[0].flatten()
    f1 = frames[2].flatten()
    f = frames[7].flatten()

    # Calculate Pearson correlation
    corr_adjacent = np.corrcoef(f0, f1)[0, 1]
    corr_jump = np.corrcoef(f0, f)[0, 1]

    print(f"Correlation adjacent frames: {corr_adjacent:.4f}")
    print(f"Correlation inteleaved frames: {corr_jump:.4f}")
    print("-" * 30)

    if corr_jump > corr_adjacent:
        print("Verdict: INTERLEAVED")
        print("Reason: Interleaved frames are structurally closer related.")
    else:
        print("Verdict: SEQUENTIAL")
        print("Reason: Adjacent frames are structurally closer related.")
        
        
import h5py

def inspect_h5_metadata(h5_path):
    def print_attrs(name, obj):
        if obj.attrs:
            print(f"\n--- Attributes for: /{name} ---")
            for key, val in obj.attrs.items():
                # Decode bytes to strings if necessary
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='ignore')
                
                val_str = str(val)
                # Truncate massively long config strings for terminal readability
                if len(val_str) > 150:
                    val_str = val_str[:150] + " ... [TRUNCATED]"
                    
                print(f"{key}: {val_str}")

    with h5py.File(h5_path, 'r') as f:
        # 1. Check the root level attributes
        if f.attrs:
            print("--- Attributes for: / (Root) ---")
            for key, val in f.attrs.items():
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='ignore')
                print(f"{key}: {str(val)[:150]}")
                
        # 2. Walk through all groups and datasets
        f.visititems(print_attrs)

# Run the test
DATA_DIR = "C:/Users/sosok/Downloads/stack_1-A1-GCaMP8M_channel_1_obj_bottom-20260630T194014Z-3-002/stack_1-A1-GCaMP8M_channel_1_obj_bottom/Cam_long_00000.lux.h5"
diagnose_h5_ordering(DATA_DIR)
# inspect_h5_metadata(DATA_DIR)
# import matplotlib.pyplot as plt

# with h5py.File(DATA_DIR, 'r') as f:
#     dataset = f['Data']
    
#     # Extract the first 3 frames along the first axis
#     frame_0 = dataset[0, :, :]
#     frame_1 = dataset[1, :, :]
#     frame_2 = dataset[2, :, :]
#     z_plane_0 = dataset[0:700, :, :]

# Plot them side-by-side
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(frame_0, cmap='gray')
# axes[0].set_title('Index 0')
# axes[1].imshow(frame_1, cmap='gray')
# axes[1].set_title('Index 1')
# axes[2].imshow(frame_2, cmap='gray')
# axes[2].set_title('Index 2')
# plt.show()