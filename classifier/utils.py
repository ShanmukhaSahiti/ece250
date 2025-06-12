import os
import glob
import numpy as np
from scipy.io import loadmat

def calculate_time_quantile(spectrogram, frequencies, q):
    """
    Calculates the time-varying q-quantile of a spectrogram.
    This implements Equation 6 from the paper.
    """
    # Normalize each time-slice of the spectrogram to get a CDF
    # Add a small epsilon to avoid division by zero for silent slices
    s_norm = spectrogram / (np.sum(spectrogram, axis=0) + 1e-9)
    s_cdf = np.cumsum(s_norm, axis=0)
    
    # Find the index where the CDF exceeds the quantile q for each time step
    q_indices = np.argmax(s_cdf >= q, axis=0)
    
    # Map the indices to corresponding frequency values
    time_quantiles = frequencies[q_indices]
    
    return time_quantiles

def extract_repetition_features(spectrograms, time_vector, freq_vector):
    """
    Extracts the 33 features described in the paper from one repetition.
    """
    all_features = []
    
    # 1. Quantile Histogram Features (3 links * 2 quantiles * 5 bins = 30 features)
    quantile_bins = [0, 10, 20, 30, 40, 100]
    quantiles_to_calc = [0.5, 0.7]
    
    quantile_curves = {} # To store for asymmetry calculation

    for i in range(3): # For each link
        sp = spectrograms[i]
        
        for q in quantiles_to_calc:
            # Calculate the time-varying quantile curve
            q_curve = calculate_time_quantile(sp, freq_vector, q)
            
            if i == 2: # Link 3
                quantile_curves[q] = q_curve

            # Calculate the 5-bin histogram for this curve
            hist, _ = np.histogram(q_curve, bins=quantile_bins)
            all_features.extend(hist)

    # 2. Temporal Asymmetry Features (Link 3 only, 2 features)
    for q in quantiles_to_calc:
        q_curve = quantile_curves[q]
        
        # Split the repetition into two halves
        mid_point = len(q_curve) // 2
        first_half_max = np.max(q_curve[:mid_point]) if mid_point > 0 else 0
        second_half_max = np.max(q_curve[mid_point:]) if mid_point < len(q_curve) else 0
        
        # Binary feature: is the difference > 10 Hz?
        asymmetry_feature = 1 if abs(first_half_max - second_half_max) > 10 else 0
        all_features.append(asymmetry_feature)
        
    # 3. Repetition Duration Feature (1 feature)
    duration = time_vector[-1] - time_vector[0]
    all_features.append(duration)
    
    # Total should be 33 features
    return all_features


def load_data(data_dir):
    """Load and process all .mat files from the data directory."""
    all_features = []
    all_labels = []
    
    # Get all activity folders
    activity_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"\nFound activity folders: {activity_folders}")
    
    for activity in activity_folders:
        activity_path = os.path.join(data_dir, activity)
        mat_files = [f for f in os.listdir(activity_path) if f.endswith('.mat')]
        print(f"\nProcessing activity '{activity}': Found {len(mat_files)} .mat files.")
        
        processed_files = 0
        for mat_file in mat_files:
            file_path = os.path.join(activity_path, mat_file)
            print(f"\nDebug - File: {mat_file}")
            
            try:
                # Load the .mat file
                data = loadmat(file_path)
                print(f"Available keys: {list(data.keys())}")
                
                # Check for required keys
                required_keys = ['T', 'F', 'sp_all']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"  - WARNING: Missing required keys: {missing_keys}")
                    continue
                
                # Extract data
                T = data['T'].flatten()  # Time points
                F = data['F'].flatten()  # Frequency points
                sp_all = data['sp_all']  # Spectrogram data
                
                # The .mat files store sp_all as a cell array, which loads as an object array.
                # We need to handle this structure.
                if sp_all.dtype == 'object':
                    # If it's an object array, we assume it's a list of spectrograms (one for each link).
                    # We'll extract them into a proper 3D numpy array.
                    # Assuming dimensions are (time, freq) for each cell.
                    # Let's find the max dimensions to create a padded numpy array.
                    max_time = 0
                    max_freq = 0
                    for i in range(sp_all.shape[0]):
                         for j in range(sp_all.shape[1]):
                            if sp_all[i,j] is not None and sp_all[i,j].ndim > 0:
                                if sp_all[i,j].shape[0] > max_time:
                                    max_time = sp_all[i,j].shape[0]
                                if sp_all[i,j].shape[1] > max_freq:
                                    max_freq = sp_all[i,j].shape[1]
                    
                    if max_time == 0 or max_freq == 0:
                        print("  - WARNING: Found object array for sp_all, but it's empty.")
                        continue # Skip this file

                    # Create a new 3D array and fill it
                    num_links = sp_all.shape[0] * sp_all.shape[1]
                    proper_sp_all = np.zeros((max_time, max_freq, num_links))
                    
                    link_k = 0
                    for i in range(sp_all.shape[0]):
                        for j in range(sp_all.shape[1]):
                            if sp_all[i,j] is not None and sp_all[i,j].ndim > 0:
                                t_dim, f_dim = sp_all[i,j].shape
                                proper_sp_all[0:t_dim, 0:f_dim, link_k] = sp_all[i,j]
                            link_k += 1
                    sp_all = proper_sp_all

                print(f"  Shapes - sp_all: {sp_all.shape}, T: {T.shape}, F: {F.shape}")
                
                # Handle period_start_times - Per user request, always treat as a single repetition
                period_starts = np.array([T[0]])
                
                # Process each repetition
                for i in range(len(period_starts)):
                    start_idx = np.where(T >= period_starts[i])[0][0]
                    end_idx = len(T) # Always go to the end of the signal
                    
                    # Extract the repetition segment
                    if end_idx - start_idx < 10:  # Minimum points threshold
                        print(f"  - WARNING: Not enough points in repetition {i+1} (found {end_idx - start_idx})")
                        continue
                    
                    # Prepare list of spectrograms for feature extraction
                    spectrogram_list = []
                    if sp_all.ndim == 3:
                        # We have multiple links, split them
                        for link_idx in range(sp_all.shape[2]):
                            spectrogram_list.append(sp_all[start_idx:end_idx, :, link_idx])
                    else:
                        # We have a single 2D spectrogram
                        sp_segment = sp_all[start_idx:end_idx, :]
                        # The feature extractor expects 3 links, so we can duplicate the single one
                        spectrogram_list = [sp_segment] * 3

                    # Extract features
                    features = extract_repetition_features(spectrogram_list, T[start_idx:end_idx], F)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(activity)
                        print(f"  - Successfully processed repetition {i+1}")
                
                processed_files += 1
                
            except Exception as e:
                print(f"  - ERROR processing file: {str(e)}")
                continue
        
        print(f"Finished '{activity}': Successfully processed {processed_files}/{len(mat_files)} files.")
    
    if not all_features:
        raise ValueError("No valid features were extracted from any files")
    
    print("\nFinalizing data arrays...")
    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"Data loaded: {len(X)} samples, {X.shape[1]} features each.")
    
    return X, y

def load_data_from_files(file_paths):
    """Load and process a list of .mat files."""
    all_features = []
    all_labels = []
    
    for file_path in file_paths:
        try:
            # Extract activity name from path (assuming path is like .../activity_name/file.mat)
            activity = os.path.basename(os.path.dirname(file_path))
            mat_file = os.path.basename(file_path)
            print(f"\nProcessing file: {mat_file} for activity: {activity}")

            # Load the .mat file
            data = loadmat(file_path)
            
            # Check for required keys
            required_keys = ['T', 'F', 'sp_all']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                print(f"  - WARNING: Missing required keys: {missing_keys}")
                continue
            
            # Extract data
            T = data['T'].flatten()
            F = data['F'].flatten()
            sp_all = data['sp_all']
            
            # The .mat files store sp_all as a cell array, which loads as an object array.
            if sp_all.dtype == 'object':
                max_time = 0
                max_freq = 0
                for i in range(sp_all.shape[0]):
                    for j in range(sp_all.shape[1]):
                        if sp_all[i,j] is not None and sp_all[i,j].ndim > 0:
                            if sp_all[i,j].shape[0] > max_time:
                                max_time = sp_all[i,j].shape[0]
                            if sp_all[i,j].shape[1] > max_freq:
                                max_freq = sp_all[i,j].shape[1]
                
                if max_time == 0 or max_freq == 0:
                    print("  - WARNING: Found object array for sp_all, but it's empty.")
                    continue

                num_links = sp_all.shape[0] * sp_all.shape[1]
                proper_sp_all = np.zeros((max_time, max_freq, num_links))
                
                link_k = 0
                for i in range(sp_all.shape[0]):
                    for j in range(sp_all.shape[1]):
                        if sp_all[i,j] is not None and sp_all[i,j].ndim > 0:
                            t_dim, f_dim = sp_all[i,j].shape
                            proper_sp_all[0:t_dim, 0:f_dim, link_k] = sp_all[i,j]
                        link_k += 1
                sp_all = proper_sp_all

            # Always treat as a single repetition
            period_starts = np.array([T[0]])
            
            for i in range(len(period_starts)):
                start_idx = 0
                end_idx = len(T)
                
                if end_idx - start_idx < 10:
                    print(f"  - WARNING: Not enough points in signal.")
                    continue
                
                spectrogram_list = []
                if sp_all.ndim == 3:
                    for link_idx in range(sp_all.shape[2]):
                        spectrogram_list.append(sp_all[start_idx:end_idx, :, link_idx])
                else:
                    sp_segment = sp_all[start_idx:end_idx, :]
                    spectrogram_list = [sp_segment] * 3

                features = extract_repetition_features(spectrogram_list, T[start_idx:end_idx], F)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(activity)
            
        except Exception as e:
            print(f"  - ERROR processing file {file_path}: {str(e)}")
            continue
            
    if not all_features:
        # Return empty arrays if no features were extracted
        return np.array([]), np.array([])
    
    X = np.array(all_features)
    y = np.array(all_labels)
    return X, y

def extract_features(sp, F):
    """Extract features from a spectrogram segment."""
    try:
        # Ensure sp is 2D
        if sp.ndim == 3:
            sp = np.mean(sp, axis=2)  # Average across channels
        
        # Calculate quantile histograms
        quantiles_to_calc = [0.25, 0.5, 0.75]
        quantile_features = []
        for q in quantiles_to_calc:
            quantile_val = np.quantile(sp, q)
            quantile_features.append(quantile_val)
        
        # Calculate temporal asymmetry
        mid_point = sp.shape[0] // 2
        first_half_max = np.max(sp[:mid_point])
        second_half_max = np.max(sp[mid_point:])
        asymmetry_feature = 1 if abs(first_half_max - second_half_max) > 10 else 0
        
        # Calculate duration feature
        duration = sp.shape[0]
        
        # Combine all features
        features = quantile_features + [asymmetry_feature, duration]
        return features
        
    except Exception as e:
        print(f"  - ERROR extracting features: {str(e)}")
        return None 