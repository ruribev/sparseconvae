## Sparse Convolutional Autoencoder for Subsurface Reconstruction

A PyTorch implementation of a sparse convolutional autoencoder for geological subsurface reconstruction, integrating Vertical Electrical Sounding (VES) data with basement boundary modeling. This implementation is based on the methodology described in "AI-based geological subsurface reconstruction using sparse convolutional autoencoders" 

## Features

- Sparse convolutional architecture for efficient processing of irregular spatial data
- Multi-scale feature extraction with ConvNeXt-based encoder
- Dual-decoder design for simultaneous reconstruction of:
  - Primary variable (subsurface resistivity)
  - Secondary variable (VES measurements)
- Dynamic spatial sampling strategies
- Integration with ResIPy for VES forward modeling
- Transfer learning capabilities from pre-trained inverse distance models

## Installation

### Requirements
- Python 3.9+
- CUDA-compatible GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM

### Dependencies
```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch ≥2.0.0
- NumPy ≥1.21.0
- SciPy ≥1.7.0
- ResIPy ≥3.3.0

## Model Architecture

### Encoder
- Four hierarchical stages with ConvNeXt blocks
- Progressive channel expansion: 96 → 192 → 384 → 768
- Sparse convolutions for memory-efficient processing
- Layer normalization and GELU activation

### Decoders
Two parallel decoders with identical architectures but independent weights:
1. **Primary Decoder**: Reconstructs subsurface resistivity
2. **Secondary Decoder**: Reconstructs VES measurements

Each decoder features:
- Four inverse stages mirroring encoder structure
- UNet-style skip connections
- Transposed convolutions for upsampling
- ReLU6 activations and batch normalization

## Usage

### Basic Example
```python
from sparseconvae.model import SparseConvAutoencoder
from sparseconvae.utils import create_sparse_tensor

# Initialize model
model = SparseConvAutoencoder(
    input_channels=1,
    encoder_channels=[96, 192, 384, 768],
    grid_size=(64, 32)
)

# Prepare data
spatial_coords = [(x1, y1), (x2, y2), ...]  # Sampling locations
values = [v1, v2, ...]  # Measured values
sparse_input = create_sparse_tensor(spatial_coords, values, grid_size=(64, 32))

# Forward pass
resistivity, ves_response = model(sparse_input)
```

### VES Integration Example
```python
from sparseconvae.ves import VESForwardModel
from sparseconvae.data import prepare_ves_data

# Setup VES forward modeling
ves_model = VESForwardModel(
    domain_size=(16250, 8125),
    electrode_configs='schlumberger'
)

# Process VES data
ves_data = prepare_ves_data(
    stations=41,  # Number of VES stations
    spacings=32   # Electrode spacings per station
)

# Combine with autoencoder
reconstructed_section = model.reconstruct_with_ves(
    ves_data=ves_data,
    ves_model=ves_model
)
```

### Training Example
```python
from sparseconvae.training import Trainer

trainer = Trainer(
    model=model,
    learning_rate=1e-4,
    weight_decay=1e-5,
    batch_size=16
)

# Train with dynamic sampling
trainer.train(
    training_data=dataset,
    epochs=320,
    sampling_strategy='dynamic',
    sampling_density=0.05  # 5% sampling
)
```

## Data Preparation

### Spatial Data Format
- Input coordinates should be normalized to [0,1] range
- Values should be scaled appropriately for the model
- Sparse tensor conversion handles irregular sampling automatically

### VES Data Requirements
- Schlumberger array configurations
- AB spacings: 60-1250 meters
- MN spacings: 15m, 50m, 150m depending on AB spacing
- Measurements organized as 64x32 arrays per profile

## Performance Metrics

Based on validation tests:
- Kriging emulation: MSE 1.2×10⁻³ (12,800 training examples)
- Subsurface reconstruction: 37.4-61.7% improvement over kriging
- VES integration: Error reduction from 4.1×10⁻¹ to 9.1×10⁻³ with 40 stations

## Citation

Paper Under Review

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines in CONTRIBUTING.md.

## Contact

- Rodrigo Uribe-Ventura - a20234215@pucp.edu.pe
- Grupo de Investigación en Geología Sedimentaria, PUCP
