import jax
import jax.numpy as np
from neural_tangents import stax

# Define WideResnet block with LayerNorm
def WideResnet(block_size=4, k=5, num_classes=10, activation_fn=stax.Relu()):
    """Constructs a WideResNet with a specified block size and width multiplier (k)."""
    width = 16 * k
    depth = block_size * 6 + 4
    
    # Define layers
    layers = []
    layers += [stax.Conv(width, (3, 3), (1, 1), padding="SAME"), activation_fn]
    
    for _ in range(block_size):
        layers += [
            stax.Conv(width, (3, 3), padding="SAME"),
            stax.LayerNorm(),
            activation_fn,
            stax.Conv(width, (3, 3), padding="SAME"),
            stax.LayerNorm()
        ]
        
    layers += [stax.GlobalAvgPool(), stax.Dense(num_classes)]
    
    # Stack layers to create WideResnet model
    return stax.serial(*layers)

# Define ConvNeXt block
def ConvNeXt(num_classes=10):
    """Defines a ConvNeXt style architecture for use in NTK experiments."""
    return stax.serial(
        stax.Conv(96, (3, 3), padding="SAME"),
        stax.Relu(),
        stax.Conv(192, (3, 3), padding="SAME"),
        stax.Relu(),
        stax.GlobalAvgPool(),
        stax.Dense(num_classes)
    )

def SimpleCNN():
    # Define each layer of the CNN model using stax
    layers = stax.serial(
        stax.Conv(32, (3, 3), (1, 1), padding="SAME"),
        stax.Relu(),
        stax.Conv(64, (3, 3), (1, 1), padding="SAME"),
        stax.Relu(),
        stax.Flatten,
        stax.Dense(128),
        stax.Relu(),
        stax.Dense(10)  # Adjust based on number of classes
    )
    
    # Unpack the layers into init_fn, apply_fn, kernel_fn
    init_fn, apply_fn, kernel_fn = layers
    return init_fn, apply_fn, kernel_fn