import torch

sample_rate = 16000
win_length = int(0.025 * sample_rate)  # 25ms window
win_hop = int(0.010 * sample_rate)     # 10ms hop
fft_length = 512
window = torch.hann_window(win_length)
        
# Filterbank parameters
nfilt = 32
bandmode = "fbank"  # or "erb" or "mfcc"
        
# Model architecture
batch_size = 4
n_expand_fbank = 1
freq_cut = 128      # Frequency cutoff point for low/high split
mask_mode = "C"     # Complex mask mode
        
# Training parameters
learning_rate = 1e-4
weight_decay = 1e-5
