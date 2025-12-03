End-to-End Lens Design
======================

Joint optimization of optical systems and deep neural networks for task-specific imaging.

Overview
--------

End-to-end design co-optimizes:

1. **Optical System**: Lens parameters (radii, thicknesses, aspherics)
2. **Neural Network**: Image reconstruction or processing network
3. **Task Objective**: Final application metric (e.g., image quality, classification accuracy)

This approach can produce optical designs specifically tailored for the target application.

Example: Lens-Network Co-Design
--------------------------------

Step 1: Setup
^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.optim as optim
    from deeplens import GeoLens
    from deeplens.network import UNet
    from torch.utils.data import DataLoader
    
    # Create lens
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Initialize constraints for optimization
    lens.init_constraints()
    
    # Create reconstruction network
    network = UNet(
        in_channels=3,
        out_channels=3
    ).cuda()

Step 2: Data Loading
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torchvision import datasets, transforms
    
    # Training dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(
        root='./datasets/BSDS300/images/train',
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

Step 3: Joint Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Separate optimizers for lens and network
    # Use lens.get_optimizer_params() to get properly configured lens parameters
    lens_params = lens.get_optimizer_params(
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],  # [d, c, k, a]
        decay=0.01
    )
    optimizer_lens = optim.Adam(lens_params)
    optimizer_net = optim.Adam(network.parameters(), lr=1e-4)
    
    # Loss functions
    from deeplens.network import SSIMLoss
    ssim_loss = SSIMLoss()
    mse_loss = torch.nn.MSELoss()  # Use PyTorch's MSELoss
    
    # Training loop
    from deeplens.basics import DEPTH, SPP_RENDER
    num_epochs = 100
    depth = DEPTH  # -20000.0 mm default
    
    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.cuda()
            
            # ===== Forward Pass =====
            # 1. Render through lens using ray tracing
            images_degraded = lens.render(
                images,
                depth=depth,
                method='ray_tracing',  # Options: 'ray_tracing', 'psf_map', 'psf_patch'
                spp=SPP_RENDER         # Samples per pixel (32 default)
            )
            
            # 2. Reconstruct with network
            images_restored = network(images_degraded)
            
            # ===== Loss Calculation =====
            # Image reconstruction loss
            loss_img = mse_loss(images_restored, images)
            loss_img += 0.5 * (1.0 - ssim_loss(images_restored, images))
            
            # Lens regularization constraints
            loss_reg, loss_dict = lens.loss_reg()
            
            # Total loss
            loss = loss_img + 0.05 * loss_reg
            
            # ===== Backward Pass =====
            optimizer_lens.zero_grad()
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_lens.step()
            optimizer_net.step()
            
            # ===== Logging =====
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Image Loss: {loss_img.item():.6f}")
                print(f"  Reg Loss: {loss_reg.item():.6f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'network_state': network.state_dict(),
            }, f'checkpoint_epoch{epoch}.pth')
            lens.write_lens_json(f'lens_epoch{epoch}.json')

Step 4: Evaluation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import batch_psnr, batch_ssim
    from deeplens.basics import SPP_RENDER
    
    network.eval()
    
    # Test dataset
    test_dataset = datasets.ImageFolder(
        root='./datasets/BSDS300/images/test',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.cuda()
            
            # Forward (use higher spp for evaluation)
            images_degraded = lens.render(
                images, 
                depth=depth, 
                method='ray_tracing',
                spp=64  # More samples for better quality
            )
            images_restored = network(images_degraded)
            
            # Metrics
            psnr = batch_psnr(images_restored, images)
            ssim = batch_ssim(images_restored, images)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    print(f"\\nTest Results:")
    print(f"  Average PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB")
    print(f"  Average SSIM: {sum(ssim_values)/len(ssim_values):.4f}")

Running the Example
-------------------

.. code-block:: bash

    python 1_end2end_lens_design.py

With configuration:

.. code-block:: bash

    python 1_end2end_lens_design.py --config configs/1_end2end_lens_design.yml

Example Configuration
---------------------

``configs/1_end2end_lens_design.yml``:

.. code-block:: yaml

    lens:
      filename: './datasets/lenses/camera/ef50mm_f1.8.json'
      optimize_params:
        radius: true
        thickness: true
        ai: true
    
    network:
      type: 'UNet'
      in_channels: 3
      out_channels: 3
      base_channels: 64
    
    training:
      num_epochs: 100
      batch_size: 4
      learning_rate_lens: 0.001
      learning_rate_network: 0.0001
      depth: 1000.0
      spp: 256
    
    data:
      train_dir: './datasets/BSDS300/images/train'
      test_dir: './datasets/BSDS300/images/test'
      image_size: [512, 512]

Task-Specific Design
--------------------

Image Classification
^^^^^^^^^^^^^^^^^^^^

Optimize lens for image classification accuracy:

.. code-block:: python

    import torchvision.models as models
    from deeplens.basics import DEPTH, SPP_RENDER
    
    # Pre-trained classifier
    classifier = models.resnet18(weights='IMAGENET1K_V1').cuda()
    classifier.eval()  # Freeze classifier
    
    # Optimize only lens
    lens_params = lens.get_optimizer_params(lrs=[1e-4, 1e-4, 0, 0])
    optimizer = optim.Adam(lens_params)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(100):
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            # Render through lens
            images_rendered = lens.render(
                images, 
                depth=DEPTH,
                method='ray_tracing',
                spp=SPP_RENDER
            )
            
            # Classify
            outputs = classifier(images_rendered)
            loss = criterion(outputs, labels)
            
            # Add lens regularization
            loss_reg, _ = lens.loss_reg()
            loss = loss + 0.01 * loss_reg
            
            # Optimize lens
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

See ``4_tasklens_img_classi.py`` for complete example.

Object Detection
^^^^^^^^^^^^^^^^

.. code-block:: python

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    # Detection model
    detector = fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    detector.eval()
    
    # Custom loss for detection performance
    def detection_loss(predictions, targets):
        # Implementation depends on detection metric
        # E.g., mAP, IoU, etc.
        pass

Depth Estimation
^^^^^^^^^^^^^^^^

.. code-block:: python

    from depth_estimation_model import DepthEstimator
    
    depth_model = DepthEstimator().cuda()
    
    # Optimize for depth estimation accuracy
    for images, depth_gt in dataloader:
        images_rendered = lens.render(images, depth=depth)
        depth_pred = depth_model(images_rendered)
        loss = depth_loss(depth_pred, depth_gt)

Advanced Techniques
-------------------

Alternating Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Alternate between lens and network optimization:

.. code-block:: python

    from deeplens.basics import DEPTH, SPP_RENDER
    
    for epoch in range(100):
        # Phase 1: Optimize network (freeze lens)
        for _ in range(5):
            images_degraded = lens.render(images, depth=DEPTH, method='ray_tracing', spp=SPP_RENDER)
            images_restored = network(images_degraded)
            loss = mse_loss(images_restored, images)
            
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
        
        # Phase 2: Optimize lens (freeze network)
        for _ in range(1):
            images_degraded = lens.render(images, depth=DEPTH, method='ray_tracing', spp=SPP_RENDER)
            images_restored = network(images_degraded)
            loss = mse_loss(images_restored, images)
            
            # Add lens regularization
            loss_reg, _ = lens.loss_reg()
            loss = loss + 0.05 * loss_reg
            
            optimizer_lens.zero_grad()
            loss.backward()
            optimizer_lens.step()

Multi-Depth Training
^^^^^^^^^^^^^^^^^^^^

Train across multiple object distances:

.. code-block:: python

    from deeplens.basics import SPP_RENDER
    
    depths = [-500.0, -1000.0, -2000.0, -5000.0]  # Negative values (object in front of lens)
    
    loss = 0.0
    for depth in depths:
        images_degraded = lens.render(images, depth=depth, method='ray_tracing', spp=SPP_RENDER)
        images_restored = network(images_degraded)
        loss += mse_loss(images_restored, images)

Perceptual Loss
^^^^^^^^^^^^^^^

Use perceptual loss for better visual quality:

.. code-block:: python

    from deeplens.network import PerceptualLoss
    
    perceptual_loss = PerceptualLoss(model='vgg19').cuda()
    
    loss = 0.5 * mse_loss(restored, target) + \
           0.5 * perceptual_loss(restored, target)

Tips and Best Practices
------------------------

1. **Learning Rates**: Use lower LR for lens (1e-3 to 1e-4) than network (1e-4 to 1e-5)
2. **Initialization**: Start with good initial lens design
3. **Constraints**: Always include lens constraints for physical realizability
4. **Pretrained Networks**: Use pretrained networks when possible
5. **Batch Size**: Smaller batches for memory efficiency
6. **SPP**: Balance speed vs accuracy (256-512 for training, 1024+ for eval)
7. **Validation**: Regularly evaluate on held-out test set
8. **Visualization**: Monitor both optical and image metrics

Expected Results
----------------

Compared to fixed optics + network:

* **Better Task Performance**: 2-5% improvement in classification accuracy
* **Simpler Optics**: Fewer elements or relaxed tolerances
* **Novel Designs**: Non-intuitive optical solutions
* **Application-Specific**: Tailored to specific imaging conditions

Limitations
-----------

* **Fabrication**: Optimized designs must be manufacturable
* **Generalization**: May overfit to training distribution
* **Computational Cost**: Requires significant GPU memory and time
* **Local Minima**: May not find global optimum

Comparison with Traditional Design
-----------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - Traditional Design
     - End-to-End Design
   * - **Approach**
     - Optimize optics, then add processing
     - Joint optimization
   * - **Objective**
     - Optical metrics (MTF, spot size)
     - Task performance
   * - **Flexibility**
     - General purpose
     - Application-specific
   * - **Design Time**
     - Weeks to months
     - Days (with GPU)
   * - **Innovation**
     - Based on experience
     - Data-driven discovery

See Also
--------

* :doc:`automated_lens_design` - Pure optical optimization
* :doc:`../tutorials` - Detailed tutorials
* :doc:`../user_guide/neural_networks` - Network architectures
* Paper: `Nature Communications 2024 <https://www.nature.com/articles/s41467-024-50835-7>`_

