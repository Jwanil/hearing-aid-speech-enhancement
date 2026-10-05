import torch
from asteroid.models import ConvTasNet

def get_1d_cnn_model():
    """
    Returns the Conv-TasNet model configured for our speech enhancement task.
    
    Architecture:
    - 256 encoder filters (learned STFT)
    - 3 TCN repeat cycles (dilated convolutions)
    - Kernel size 16, Stride 8 (controls latency: 8/16000 = 0.5ms stride latency)
    
    Returns:
        model: A PyTorch ConvTasNet model (untrained)
    """
    model = ConvTasNet(
        n_src=1,         # 1 output source (enhancement, not separation)
        n_filters=256,   # N: number of encoder filters
        n_repeats=3,     # number of TCN repeat cycles
        bn_chan=128,     # bottleneck channels
        hid_chan=256,    # hidden channels in depthwise conv
        skip_chan=128,   # skip connection channels
        kernel_size=16,  # encoder/decoder kernel size
        stride=8         # encoder/decoder stride
    )
    return model

if __name__ == "__main__":
    print("Testing 1D CNN (Conv-TasNet) Architecture...")
    model = get_1d_cnn_model()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    
    # Test forward pass with a dummy tensor (Batch=1, Channels=1, Samples=64000)
    print("Running forward pass test (1 batch, 4 seconds at 16kHz)...")
    dummy_input = torch.randn(1, 1, 64000)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert dummy_input.shape == output.shape, "Shape mismatch between input and output!"
    print("✅ Forward pass successful!")
