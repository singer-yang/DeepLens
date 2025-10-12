Neural Networks
===============

DeepLens includes neural network modules for surrogate modeling, image reconstruction, and computational imaging.

Overview
--------

The ``deeplens.network`` package contains:

* **Surrogate Models**: Fast neural approximations of optical systems
* **Reconstruction Networks**: Deep learning for image restoration
* **Loss Functions**: Perceptual and optical quality metrics

Surrogate Networks
------------------

PSFNet
^^^^^^

Neural network for fast PSF prediction across depth and field positions.

.. code-block:: python

    from deeplens.network import PSFNet
    
    # Create network
    psfnet = PSFNet(
        in_channels=3,        # [depth, field_x, field_y]
        out_channels=1,       # PSF
        hidden_dim=256,
        num_layers=8,
        psf_size=64,
        device='cuda'
    )
    
    # Forward pass
    psf = psfnet(
        depth=torch.tensor([1000.0]),
        field=torch.tensor([0.0, 0.5]),
        wavelength=torch.tensor([0.550])
    )

Architecture
""""""""""""

PSFNet uses a modified MLP with:

* Coordinate-based input encoding
* Skip connections for gradient flow
* Periodic activation functions (SIREN-like)

Training
""""""""

.. code-block:: python

    from deeplens.network import PSFDataset
    import torch.optim as optim
    
    # Create dataset
    dataset = PSFDataset(
        lens=geolens,
        num_samples=10000,
        depth_range=[500, 5000],
        field_range=[0.0, 1.0]
    )
    
    # Training loop
    optimizer = optim.Adam(psfnet.parameters(), lr=1e-4)
    
    for epoch in range(100):
        for batch in dataloader:
            depth, field, psf_gt = batch
            
            # Forward
            psf_pred = psfnet(depth, field)
            
            # Loss
            loss = torch.nn.functional.mse_loss(psf_pred, psf_gt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

See ``3_psf_net.py`` for a complete training example.

SIREN
^^^^^

Sinusoidal representation network for implicit optical representations.

.. code-block:: python

    from deeplens.network import SIREN
    
    model = SIREN(
        in_features=5,      # [x, y, depth, field_x, field_y]
        out_features=3,     # RGB PSF
        hidden_features=256,
        hidden_layers=8,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0
    )

**Key Features:**

* Periodic activation: :math:`\\sin(\\omega_0 x)`
* Better learning of high-frequency details
* Implicit representation of optical fields

MLP with Convolutions
^^^^^^^^^^^^^^^^^^^^^^

Hybrid MLP-Conv architecture for spatial-variant PSF prediction.

.. code-block:: python

    from deeplens.network import MLPConv
    
    model = MLPConv(
        spatial_dim=(64, 64),    # PSF size
        condition_dim=3,         # [depth, field_x, field_y]
        hidden_dim=512,
        num_layers=6,
        use_skip=True
    )

Modulated SIREN
^^^^^^^^^^^^^^^

SIREN with FiLM (Feature-wise Linear Modulation) conditioning.

.. code-block:: python

    from deeplens.network import ModulateSIREN
    
    model = ModulateSIREN(
        in_features=2,          # [x, y]
        condition_features=3,   # [depth, field_x, field_y]
        out_features=1,
        hidden_features=256,
        hidden_layers=8
    )

Reconstruction Networks
-----------------------

UNet
^^^^

Standard UNet for image restoration.

.. code-block:: python

    from deeplens.network import UNet
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_scales=4,
        use_dropout=False,
        device='cuda'
    )
    
    # Restore image
    restored = model(degraded_image)

**Applications:**

* Deblurring
* Denoising
* Super-resolution
* Aberration correction

NAFNet
^^^^^^

Nonlinear Activation Free Network for efficient image restoration.

.. code-block:: python

    from deeplens.network import NAFNet
    
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    )

**Advantages:**

* No nonlinear activations (faster, simpler)
* State-of-the-art restoration quality
* Memory efficient

Restormer
^^^^^^^^^

Transformer-based restoration network.

.. code-block:: python

    from deeplens.network import Restormer
    
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    )

**Features:**

* Multi-scale attention mechanism
* Global receptive field
* Excellent for large degradations

SwinIR
^^^^^^

Swin Transformer for image restoration.

.. code-block:: python

    from deeplens.network import SwinIR
    
    model = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        upscale=1
    )

Loss Functions
--------------

MSE Loss
^^^^^^^^

Standard mean squared error:

.. code-block:: python

    from deeplens.network import MSELoss
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)

PSNR Loss
^^^^^^^^^

Peak Signal-to-Noise Ratio loss:

.. code-block:: python

    from deeplens.network import PSNRLoss
    
    loss_fn = PSNRLoss()
    loss = loss_fn(pred, target)

**Note:** Minimizing negative PSNR maximizes image quality.

SSIM Loss
^^^^^^^^^

Structural Similarity Index loss:

.. code-block:: python

    from deeplens.network import SSIMLoss
    
    loss_fn = SSIMLoss(
        window_size=11,
        size_average=True
    )
    loss = loss_fn(pred, target)

Perceptual Loss
^^^^^^^^^^^^^^^

VGG-based perceptual loss:

.. code-block:: python

    from deeplens.network import PerceptualLoss
    
    loss_fn = PerceptualLoss(
        model='vgg19',
        layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'],
        weights=[1.0, 1.0, 1.0, 1.0],
        device='cuda'
    )
    loss = loss_fn(pred, target)

**Advantages:**

* Better perceptual quality
* Captures high-level features
* Less sensitive to pixel-wise shifts

Combined Loss
^^^^^^^^^^^^^

Combine multiple loss functions:

.. code-block:: python

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

Datasets
--------

PSF Dataset
^^^^^^^^^^^

Dataset for training PSF surrogate models:

.. code-block:: python

    from deeplens.network import PSFDataset
    
    dataset = PSFDataset(
        lens=geolens,
        num_samples=10000,
        depth_range=[500, 5000],
        field_range=[0.0, 1.0],
        wavelengths=[0.486, 0.550, 0.656],
        psf_size=64,
        spp=2048
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

Image Restoration Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Dataset for training restoration networks:

.. code-block:: python

    from deeplens.network import RestorationDataset
    
    dataset = RestorationDataset(
        clean_dir='./data/clean/',
        degraded_dir='./data/degraded/',
        patch_size=256,
        augmentation=True
    )

Custom Dataset
^^^^^^^^^^^^^^

Create custom datasets:

.. code-block:: python

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, lens, num_samples=1000):
            self.lens = lens
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random depth and field
            depth = torch.rand(1) * 4500 + 500
            field = torch.rand(2) * 2 - 1  # [-1, 1]
            
            # Generate PSF
            psf = self.lens.psf(depth=depth.item(), field=field.tolist())
            
            return depth, field, psf

End-to-End Training
-------------------

Joint Lens-Network Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import GeoLens
    from deeplens.network import UNet
    
    # Initialize lens and network
    lens = GeoLens(filename='initial_design.json', device='cuda')
    network = UNet(in_channels=3, out_channels=3).cuda()
    
    # Enable lens optimization
    lens.set_optimizer_params({'radius': True, 'thickness': True})
    
    # Combined optimizer
    optimizer = torch.optim.Adam([
        {'params': lens.parameters(), 'lr': 1e-3},
        {'params': network.parameters(), 'lr': 1e-4}
    ])
    
    # Training loop
    for epoch in range(100):
        for img_clean in dataloader:
            # Forward through lens
            img_degraded = lens.render(img_clean, depth=1000)
            
            # Restore with network
            img_restored = network(img_degraded)
            
            # Loss
            loss = torch.nn.functional.mse_loss(img_restored, img_clean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Task-Specific Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize lens for specific vision tasks:

.. code-block:: python

    import torchvision.models as models
    
    # Load pre-trained classifier
    classifier = models.resnet18(pretrained=True).cuda()
    classifier.eval()
    
    # Optimize lens for classification
    for epoch in range(100):
        for img, label in dataloader:
            # Render through lens
            img_rendered = lens.render(img, depth=1000)
            
            # Classify
            pred = classifier(img_rendered)
            
            # Classification loss
            loss = torch.nn.functional.cross_entropy(pred, label)
            
            # Optimize lens only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

See ``4_tasklens_img_classi.py`` for a complete example.

Training Utilities
------------------

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    
    # Cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Step decay
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Use in training
    for epoch in range(100):
        train_one_epoch()
        scheduler.step()

Early Stopping
^^^^^^^^^^^^^^

.. code-block:: python

    class EarlyStopping:
        def __init__(self, patience=10, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0
        
        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
    
    # Use in training
    early_stopping = EarlyStopping(patience=20)
    for epoch in range(1000):
        train_loss = train_one_epoch()
        val_loss = validate()
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

Checkpointing
^^^^^^^^^^^^^

.. code-block:: python

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pth')
    
    # Load checkpoint
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

Mixed Precision Training
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    for epoch in range(100):
        for data in dataloader:
            optimizer.zero_grad()
            
            # Forward with autocasting
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

Distributed Training
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Wrap model
    model = DDP(model, device_ids=[local_rank])
    
    # Distributed sampler
    sampler = torch.utils.data.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

Best Practices
--------------

Model Design
^^^^^^^^^^^^

1. **Start Simple**: Begin with smaller models, scale up if needed
2. **Validate Architecture**: Test on simple cases first
3. **Monitor Gradients**: Check for vanishing/exploding gradients
4. **Use Skip Connections**: Help with gradient flow

Training Strategy
^^^^^^^^^^^^^^^^^

1. **Data Augmentation**: Essential for generalization
2. **Batch Size**: Larger batches for stability, smaller for generalization
3. **Learning Rate**: Use learning rate schedulers
4. **Regularization**: Dropout, weight decay, early stopping

Computational Efficiency
^^^^^^^^^^^^^^^^^^^^^^^^

1. **GPU Memory**: Monitor and optimize memory usage
2. **Mixed Precision**: Use AMP for 2x speedup
3. **Data Loading**: Use multiple workers, pin memory
4. **Profiling**: Identify bottlenecks with PyTorch profiler

Pre-trained Models
------------------

DeepLens provides pre-trained models:

.. code-block:: python

    from deeplens.network import load_pretrained
    
    # Load PSFNet
    psfnet = load_pretrained('psfnet_ef50mm_f1.8')
    
    # Load restoration network
    restorer = load_pretrained('nafnet_deblur')

Available pre-trained models:

* ``psfnet_ef50mm_f1.8``: PSF network for Canon 50mm f/1.8
* ``nafnet_deblur``: NAFNet trained for deblurring
* ``unet_aberration``: UNet for aberration correction

Next Steps
----------

* See :doc:`../examples/end2end_design` for joint optimization examples
* Learn about :doc:`lens_systems` for optical system design
* Check :doc:`../tutorials` for training workflows
* Explore :doc:`../api/network` for detailed API reference

