import numpy as np

def remove_low_freq(x, w):
    w = int(w) # ensure w is an integer for filter size
    filt = np.ones(w) / w
    if x.ndim == 1 or (x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1)):
        x_flat = x.flatten()
        lf = np.convolve(x_flat, filt, mode='same')
        # Reshape lf to match original x shape if x was a 2D row/column vector
        if x.ndim == 2:
            lf = lf.reshape(x.shape)
    elif x.ndim == 2 : # 2D array, convolve along columns
        # Applying 1D convolution along each column (Matlab's conv2(filt,1, x, 'same') behavior)
        lf = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='same'), axis=0, arr=x)
    else:
        raise ValueError("Input x must be 1D or 2D for remove_low_freq")
    
    y = x - lf
    return y 