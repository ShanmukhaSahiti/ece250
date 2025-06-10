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


def load_data():
    """
    Loads .mat files, segments them into repetitions, extracts features
    for each repetition, and returns the features and labels.
    """
    # Get the absolute path to the directory containing this script (utils.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the spectrograms directory
    base_dir = os.path.join(script_dir, '..', 'simulated_spectrograms')
    
    X = []
    y = []
    
    activity_folders = [f for f in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(f)]
    
    print(f"Found activity folders: {[os.path.basename(f) for f in activity_folders]}")

    for activity_path in activity_folders:
        activity_label = os.path.basename(activity_path)
        mat_files = glob.glob(os.path.join(activity_path, '*.mat'))
        
        for mat_file in mat_files:
            try:
                data = loadmat(mat_file)
                sp_all = data['sp_all']        # Spectrograms for 3 links
                T = data['T'].flatten()        # Time vector
                F = data['F'].flatten()        # Frequency vector
                period_starts = data['period_start_times'].flatten()

                # Segment the data into repetitions
                for i in range(len(period_starts)):
                    start_time = period_starts[i]
                    end_time = period_starts[i+1] if i + 1 < len(period_starts) else T[-1]
                    
                    # Find indices corresponding to this repetition's time window
                    rep_indices = np.where((T >= start_time) & (T <= end_time))[0]
                    
                    if len(rep_indices) < 2: # Need at least 2 time steps for a repetition
                        continue
                        
                    # Extract the spectrograms for this repetition by slicing each link's spectrogram
                    rep_sps = np.array([sp_all[i, 0][:, rep_indices] for i in range(sp_all.shape[0])])
                    rep_time_vector = T[rep_indices]
                    
                    # Extract the 33 features for this repetition
                    features = extract_repetition_features(rep_sps, rep_time_vector, F)
                    X.append(features)
                    y.append(activity_label)

            except Exception as e:
                print(f"Could not process file {mat_file}: {e}")

    return np.array(X), np.array(y) 