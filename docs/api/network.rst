Network API Reference
=====================

Neural network models for surrogate modeling and image reconstruction.

Surrogate Networks
------------------

PSFNet
^^^^^^

.. py:class:: deeplens.network.PSFNet(in_channels=3, out_channels=1, hidden_dim=256, num_layers=8, psf_size=64, device='cuda')

   Neural network for PSF prediction.

   :param in_channels: Input dimension (depth, field)
   :param out_channels: Output channels (1 for mono, 3 for RGB)
   :param hidden_dim: Hidden layer dimension
   :param num_layers: Number of layers
   :param psf_size: Output PSF size
   :param device: Device

   **Methods:**

   .. py:method:: forward(depth, field, wavelength=None)

      Predict PSF.

      :param depth: Object distance [B, 1]
      :param field: Field position [B, 2]
      :param wavelength: Wavelength [B, 1] (optional)
      :return: PSF [B, C, H, W]

SIREN
^^^^^

.. py:class:: deeplens.network.SIREN(in_features, out_features, hidden_features=256, hidden_layers=8, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)

   Sinusoidal representation network.

   :param in_features: Input feature dimension
   :param out_features: Output feature dimension
   :param hidden_features: Hidden layer size
   :param hidden_layers: Number of hidden layers
   :param outermost_linear: Use linear final layer
   :param first_omega_0: First layer frequency
   :param hidden_omega_0: Hidden layer frequency

   **Methods:**

   .. py:method:: forward(coords)

      Forward pass.

      :param coords: Coordinate inputs [B, in_features]
      :return: Outputs [B, out_features]

MLPConv
^^^^^^^

.. py:class:: deeplens.network.MLPConv(spatial_dim=(64, 64), condition_dim=3, hidden_dim=512, num_layers=6, use_skip=True)

   MLP with spatial convolutions.

   :param spatial_dim: Spatial dimensions (H, W)
   :param condition_dim: Conditioning vector dimension
   :param hidden_dim: Hidden dimension
   :param num_layers: Number of layers
   :param use_skip: Use skip connections

   **Methods:**

   .. py:method:: forward(condition)

      Generate spatial output from condition.

      :param condition: Condition vector [B, condition_dim]
      :return: Spatial map [B, C, H, W]

ModulateSIREN
^^^^^^^^^^^^^

.. py:class:: deeplens.network.ModulateSIREN(in_features, condition_features, out_features, hidden_features=256, hidden_layers=8)

   SIREN with FiLM modulation.

   :param in_features: Spatial input features
   :param condition_features: Conditioning features
   :param out_features: Output features
   :param hidden_features: Hidden layer size
   :param hidden_layers: Number of layers

   **Methods:**

   .. py:method:: forward(coords, condition)

      Modulated forward pass.

      :param coords: Spatial coordinates [B, in_features]
      :param condition: Condition vector [B, condition_features]
      :return: Outputs [B, out_features]

Reconstruction Networks
-----------------------

UNet
^^^^

.. py:class:: deeplens.network.UNet(in_channels=3, out_channels=3, base_channels=64, num_scales=4, use_dropout=False, device='cuda')

   UNet for image restoration.

   :param in_channels: Input channels
   :param out_channels: Output channels
   :param base_channels: Base channel count
   :param num_scales: Number of scales
   :param use_dropout: Use dropout
   :param device: Device

   **Methods:**

   .. py:method:: forward(x)

      Restore image.

      :param x: Input image [B, C, H, W]
      :return: Restored image [B, C, H, W]

NAFNet
^^^^^^

.. py:class:: deeplens.network.NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])

   Nonlinear Activation Free Network.

   :param img_channel: Image channels
   :param width: Base width
   :param middle_blk_num: Middle blocks
   :param enc_blk_nums: Encoder blocks per scale
   :param dec_blk_nums: Decoder blocks per scale

   **Methods:**

   .. py:method:: forward(inp)

      Restore image.

      :param inp: Input image [B, C, H, W]
      :return: Restored image [B, C, H, W]

Restormer
^^^^^^^^^

.. py:class:: deeplens.network.Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8], num_heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False)

   Transformer-based restoration.

   :param inp_channels: Input channels
   :param out_channels: Output channels
   :param dim: Base dimension
   :param num_blocks: Blocks per scale
   :param num_heads: Attention heads per scale
   :param ffn_expansion_factor: FFN expansion
   :param bias: Use bias

   **Methods:**

   .. py:method:: forward(inp_img)

      Restore image.

      :param inp_img: Input [B, C, H, W]
      :return: Restored [B, C, H, W]

SwinIR
^^^^^^

.. py:class:: deeplens.network.SwinIR(img_size=64, patch_size=1, in_chans=3, embed_dim=180, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6], window_size=8, upscale=1)

   Swin Transformer for restoration.

   :param img_size: Input image size
   :param patch_size: Patch size
   :param in_chans: Input channels
   :param embed_dim: Embedding dimension
   :param depths: Depth per stage
   :param num_heads: Heads per stage
   :param window_size: Window size
   :param upscale: Upscaling factor

   **Methods:**

   .. py:method:: forward(x)

      Process image.

      :param x: Input [B, C, H, W]
      :return: Output [B, C, H', W']

Loss Functions
--------------

MSELoss
^^^^^^^

.. py:class:: deeplens.network.MSELoss()

   Mean squared error loss.

   .. py:method:: forward(pred, target)

      Compute MSE.

      :param pred: Predictions [B, C, H, W]
      :param target: Ground truth [B, C, H, W]
      :return: Loss scalar

PSNRLoss
^^^^^^^^

.. py:class:: deeplens.network.PSNRLoss(max_val=1.0)

   PSNR-based loss (negative PSNR).

   :param max_val: Maximum pixel value

   .. py:method:: forward(pred, target)

      Compute PSNR loss.

      :param pred: Predictions
      :param target: Ground truth
      :return: Negative PSNR

SSIMLoss
^^^^^^^^

.. py:class:: deeplens.network.SSIMLoss(window_size=11, size_average=True)

   SSIM-based loss.

   :param window_size: Window size
   :param size_average: Average over batch

   .. py:method:: forward(pred, target)

      Compute SSIM loss (1 - SSIM).

      :param pred: Predictions
      :param target: Ground truth
      :return: Loss scalar

PerceptualLoss
^^^^^^^^^^^^^^

.. py:class:: deeplens.network.PerceptualLoss(model='vgg19', layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'], weights=[1.0, 1.0, 1.0, 1.0], device='cuda')

   VGG-based perceptual loss.

   :param model: 'vgg16' or 'vgg19'
   :param layers: Layers to use
   :param weights: Layer weights
   :param device: Device

   .. py:method:: forward(pred, target)

      Compute perceptual loss.

      :param pred: Predictions [B, 3, H, W]
      :param target: Ground truth [B, 3, H, W]
      :return: Loss scalar

Datasets
--------

PSFDataset
^^^^^^^^^^

.. py:class:: deeplens.network.PSFDataset(lens, num_samples=10000, depth_range=[500, 5000], field_range=[0.0, 1.0], wavelengths=[0.486, 0.550, 0.656], psf_size=64, spp=2048)

   Dataset for PSF surrogate training.

   :param lens: GeoLens object
   :param num_samples: Number of samples
   :param depth_range: [min, max] depth in mm
   :param field_range: [min, max] normalized field
   :param wavelengths: List of wavelengths in μm
   :param psf_size: PSF image size
   :param spp: Samples per PSF

   **Methods:**

   .. py:method:: __getitem__(idx)

      Get sample.

      :return: Tuple (depth, field, wavelength, psf)

   .. py:method:: __len__()

      Dataset length.

      :return: Number of samples

RestorationDataset
^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.network.RestorationDataset(clean_dir, degraded_dir, patch_size=256, augmentation=True)

   Dataset for image restoration training.

   :param clean_dir: Directory with clean images
   :param degraded_dir: Directory with degraded images
   :param patch_size: Patch size for training
   :param augmentation: Use data augmentation

   **Methods:**

   .. py:method:: __getitem__(idx)

      Get image pair.

      :return: Tuple (degraded, clean)

Utilities
---------

load_pretrained
^^^^^^^^^^^^^^^

.. py:function:: deeplens.network.load_pretrained(model_name, device='cuda')

   Load pre-trained model.

   :param model_name: Model identifier
   :param device: Device
   :return: Loaded model

   **Available Models:**

   * ``'psfnet_ef50mm_f1.8'``: PSF network for Canon 50mm f/1.8
   * ``'nafnet_deblur'``: NAFNet for deblurring
   * ``'unet_aberration'``: UNet for aberration correction

save_checkpoint
^^^^^^^^^^^^^^^

.. py:function:: deeplens.network.save_checkpoint(model, optimizer, epoch, loss, filename)

   Save training checkpoint.

   :param model: Model to save
   :param optimizer: Optimizer state
   :param epoch: Current epoch
   :param loss: Current loss
   :param filename: Output file path

load_checkpoint
^^^^^^^^^^^^^^^

.. py:function:: deeplens.network.load_checkpoint(filename, model, optimizer=None)

   Load training checkpoint.

   :param filename: Checkpoint file
   :param model: Model to load into
   :param optimizer: Optimizer to load into (optional)
   :return: Dictionary with epoch, loss info

Examples
--------

Training PSFNet
^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import GeoLens
    from deeplens.network import PSFNet, PSFDataset
    import torch
    from torch.utils.data import DataLoader
    
    # Create lens and network
    lens = GeoLens(filename='lens.json')
    model = PSFNet(in_channels=3, out_channels=1, psf_size=64).cuda()
    
    # Create dataset
    dataset = PSFDataset(lens, num_samples=10000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(100):
        for depth, field, wvln, psf_gt in loader:
            optimizer.zero_grad()
            psf_pred = model(depth, field, wvln)
            loss = torch.nn.functional.mse_loss(psf_pred, psf_gt)
            loss.backward()
            optimizer.step()

Using Pre-trained Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.network import load_pretrained
    
    # Load pre-trained PSFNet
    model = load_pretrained('psfnet_ef50mm_f1.8')
    
    # Predict PSF
    psf = model(depth=torch.tensor([[1000.0]]), 
                field=torch.tensor([[0.0, 0.5]]))

Image Restoration
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.network import UNet
    
    # Create model
    model = UNet(in_channels=3, out_channels=3).cuda()
    
    # Restore image
    degraded = load_image('degraded.png')
    restored = model(degraded)
    save_image(restored, 'restored.png')

Combined Loss
^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.network import MSELoss, SSIMLoss, PerceptualLoss
    
    class CombinedLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = MSELoss()
            self.ssim = SSIMLoss()
            self.perceptual = PerceptualLoss()
        
        def forward(self, pred, target):
            loss = 0.0
            loss += 1.0 * self.mse(pred, target)
            loss += 0.5 * (1.0 - self.ssim(pred, target))
            loss += 0.1 * self.perceptual(pred, target)
            return loss
    
    # Use in training
    criterion = CombinedLoss()
    loss = criterion(predicted, target)

See Also
--------

* :doc:`../user_guide/neural_networks` - Detailed network guide
* :doc:`../tutorials` - Training tutorials
* :doc:`../examples/end2end_design` - End-to-end optimization

