Network API Reference
=====================

DeepLens provides neural network modules for surrogate modeling of optics and for image reconstruction/restoration.

Surrogate Networks
------------------

MLP
^^^

.. py:class:: deeplens.network.MLP(in_features, out_features, hidden_features=64, hidden_layers=3)

   Multi-layer perceptron producing a normalized output vector (useful for PSF channel normalization).

   :param in_features: Input feature dimension
   :param out_features: Output feature dimension
   :param hidden_features: Hidden layer width
   :param hidden_layers: Number of additional hidden layers

   **Methods:**

   .. py:method:: forward(x)

      Forward pass.

      :param x: Input features [B, in_features]
      :return: Normalized outputs [B, out_features]

MLPConv
^^^^^^^

.. py:class:: deeplens.network.MLPConv(in_features, ks, channels=3, activation='relu')

   MLP encoder + convolutional decoder that maps a condition vector to a spatial kernel/map. Useful for high-frequency PSF prediction.

   :param in_features: Input feature dimension (e.g., condition) 
   :param ks: Output spatial size (kernel size)
   :param channels: Number of output channels
   :param activation: Activation in bottleneck ("relu" or "sigmoid")

   **Methods:**

   .. py:method:: forward(x)

      Generate spatial output.

      :param x: Condition vector [B, in_features]
      :return: Spatial map [B, channels, ks, ks]

Siren (sine linear layer)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.network.Siren(dim_in, dim_out, w0=1.0, c=6.0, is_first=False, use_bias=True, activation=None)

   Sine-activated linear layer used to build SIREN-style implicit networks.

   :param dim_in: Input dimension
   :param dim_out: Output dimension
   :param w0: Sine frequency multiplier
   :param c: Initialization constant
   :param is_first: Whether this is the first SIREN layer
   :param use_bias: Use bias term
   :param activation: Optional activation module (defaults to sine)

   **Methods:**

   .. py:method:: forward(x)

      :param x: Inputs [B, dim_in]
      :return: Outputs [B, dim_out]

ModulateSiren
^^^^^^^^^^^^^

.. py:class:: deeplens.network.ModulateSiren(dim_in, dim_hidden, dim_out, dim_latent, num_layers, image_width, image_height, w0=1.0, w0_initial=30.0, use_bias=True, final_activation=None, outermost_linear=True)

   SIREN-based synthesizer modulated by a latent vector (FiLM-like). Internally samples a fixed 
   coordinate grid in [-1, 1] for the given spatial size and produces an image/map.

   :param dim_in: Coordinate input dimension (typically 2)
   :param dim_hidden: Hidden width in synthesizer SIREN
   :param dim_out: Number of output channels
   :param dim_latent: Latent vector dimension used by the modulator
   :param num_layers: Number of SIREN layers in synthesizer/modulator
   :param image_width: Output width
   :param image_height: Output height
   :param w0: Frequency scale for hidden SIREN layers
   :param w0_initial: Frequency scale for the first SIREN layer
   :param use_bias: Use bias terms
   :param final_activation: Optional final activation
   :param outermost_linear: If True, ends with Linear; otherwise another SIREN layer

   **Methods:**

   .. py:method:: forward(latent)

      :param latent: Latent vector [B, dim_latent]
      :return: Outputs [B, dim_out, image_height, image_width]

PSFNet_MLPConv
^^^^^^^^^^^^^^

.. py:class:: deeplens.network.surrogate.PSFNet_MLPConv(in_chan=2, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16)

   Combined MLP conditioner and convolutional decoder for spatially varying PSF modeling.

   :param in_chan: Conditioner input dimension (e.g., 2 for (r, z))
   :param kernel_size: Output PSF size (assumes powers of 2 per implementation)
   :param out_chan: Output channels
   :param latent_dim: Conditioner output (flatten) dimension
   :param latent_channels: Channels used when reshaping latent to a feature map

   **Methods:**

   .. py:method:: forward(x)

      :param x: Conditioner inputs [B, in_chan]
      :return: PSF tensor [B, out_chan, kernel_size, kernel_size]

Reconstruction Networks
-----------------------

UNet
^^^^

.. py:class:: deeplens.network.UNet(in_channels=3, out_channels=3)

   Lightweight UNet variant for image restoration.

   :param in_channels: Input channels
   :param out_channels: Output channels

   **Methods:**

   .. py:method:: forward(x)

      :param x: Input image [B, C, H, W]
      :return: Restored image [B, C, H, W]

NAFNet
^^^^^^

.. py:class:: deeplens.network.NAFNet(in_chan=3, out_chan=3, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])

   Nonlinear Activation Free Network for image restoration.

   :param in_chan: Input channels
   :param out_chan: Output channels
   :param width: Base width
   :param middle_blk_num: Number of middle blocks
   :param enc_blk_nums: Encoder blocks per scale
   :param dec_blk_nums: Decoder blocks per scale

   **Methods:**

   .. py:method:: forward(inp)

      :param inp: Input image [B, C, H, W]
      :return: Restored image [B, C, H, W]

Restormer
^^^^^^^^^

.. py:class:: deeplens.network.Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)

   Transformer-based restoration model.

   :param inp_channels: Input channels
   :param out_channels: Output channels
   :param dim: Base dimension
   :param num_blocks: Blocks per encoder/decoder stage
   :param num_refinement_blocks: Refinement blocks at the end
   :param heads: Attention heads per scale
   :param ffn_expansion_factor: FFN expansion factor
   :param bias: Whether to use bias
   :param LayerNorm_type: 'WithBias' or 'BiasFree'
   :param dual_pixel_task: Enable dual-pixel defocus deblurring setting

   **Methods:**

   .. py:method:: forward(inp_img)

      :param inp_img: Input image [B, C, H, W]
      :return: Restored image [B, C, H, W]

SwinIR
^^^^^^

.. py:class:: deeplens.network.reconstruction.SwinIR(img_size=64, patch_size=1, in_chans=3, embed_dim=96, depths=[6,6,6,6], num_heads=[6,6,6,6], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.0, upsampler='', resi_connection='1conv')

   Swin Transformer-based restoration/upsampling model.

   :param img_size: Input image size (int or tuple)
   :param patch_size: Patch size
   :param in_chans: Input channels
   :param embed_dim: Embedding dimension
   :param depths: Depth per stage
   :param num_heads: Attention heads per stage
   :param window_size: Window size
   :param mlp_ratio: MLP ratio
   :param upscale: Upscale factor (2/3/4/8; 1 for denoising/JPEG)

Loss Functions
--------------

PSNRLoss
^^^^^^^^

.. py:class:: deeplens.network.PSNRLoss(loss_weight=1.0, reduction='mean', toY=False)

   PSNR-based loss; larger PSNR corresponds to lower loss.

   **Methods:**

   .. py:method:: forward(pred, target)

      :param pred: Predicted images [B, C, H, W]
      :param target: Ground truth images [B, C, H, W]
      :return: Loss scalar

SSIMLoss
^^^^^^^^

.. py:class:: deeplens.network.SSIMLoss(window_size=11, size_average=True)

   SSIM-based loss returning 1 - SSIM.

   **Methods:**

   .. py:method:: forward(pred, target)

      :param pred: Predicted images [B, C, H, W]
      :param target: Ground truth images [B, C, H, W]
      :return: Loss scalar (1 - SSIM)

PerceptualLoss
^^^^^^^^^^^^^^

.. py:class:: deeplens.network.PerceptualLoss(device=None, weights=[1.0, 1.0, 1.0, 1.0, 1.0])

   VGG16 feature-based perceptual loss using layers ``relu1_2``, ``relu2_2``, ``relu3_3``, ``relu4_3``, ``relu5_3``.

   **Methods:**

   .. py:method:: forward(x, y)

      :param x: Predicted images [B, 3, H, W]
      :param y: Target images [B, 3, H, W]
      :return: Perceptual loss scalar

Datasets
--------

ImageDataset
^^^^^^^^^^^^

.. py:class:: deeplens.network.ImageDataset(img_dir, img_res=None)

   Basic image dataset loader with augmentation and ImageNet-style normalization.

   **Methods:**

   .. py:method:: __getitem__(idx)

      :return: Tensor image [C, H, W]

   .. py:method:: __len__()

      :return: Number of images

PhotographicDataset
^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.network.PhotographicDataset(img_dir, img_res=(512, 512), iso_range=(100, 400), is_train=True)

   Image dataset that also samples ISO and field center for simulation-driven training.

   **Methods:**

   .. py:method:: __getitem__(idx)

      :return: Dict with keys ``img`` (Tensor [C, H, W]), ``iso`` (float), ``iso_scale`` (int), ``field_center`` (Tensor [2])

   .. py:method:: __len__()

      :return: Number of images

Dataset download helpers
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.network.dataset.download_bsd300(destination_folder='./datasets')

   Download BSDS300 to ``destination_folder`` and return the image directory path.

.. py:function:: deeplens.network.dataset.download_div2k(destination_folder)

   Download and extract DIV2K (HR) splits into ``destination_folder``.

.. py:function:: deeplens.network.dataset.download_flick2k(destination_folder='./datasets')

   Download and extract Flickr2K into ``destination_folder`` (via Hugging Face).

.. py:function:: deeplens.network.dataset.download_div8k(destination_folder='./datasets')

   Download and extract DIV8K into ``destination_folder`` (via Hugging Face).

Examples
--------

PSF modeling with PSFNet_MLPConv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.network.surrogate import PSFNet_MLPConv

    # (r, z) conditioner -> PSF
    model = PSFNet_MLPConv(in_chan=2, kernel_size=128, out_chan=3)
    rz = torch.tensor([[0.5, -5000.0], [-0.3, -2000.0]])  # [B=2, 2]
    psf = model(rz)  # [2, 3, 128, 128]

Image Restoration with NAFNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.network import NAFNet

    model = NAFNet(in_chan=3, out_chan=3)
    degraded = torch.rand(1, 3, 256, 256)
    restored = model(degraded)

Combined Loss
^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.network import PSNRLoss, SSIMLoss, PerceptualLoss

    class CombinedLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.psnr = PSNRLoss()
            self.ssim = SSIMLoss()
            self.perc = PerceptualLoss()
        def forward(self, pred, target):
            loss = 0.0
            # Lower MSE -> higher PSNR -> lower PSNRLoss
            loss += 1.0 * self.psnr(pred, target)
            # SSIMLoss already returns (1 - SSIM)
            loss += 0.5 * self.ssim(pred, target)
            loss += 0.1 * self.perc(pred, target)
            return loss

See Also
--------

* :doc:`../user_guide/neural_networks` – Overview and guidance
* :doc:`../tutorials` – Training tutorials
* :doc:`../examples/end2end_design` – End-to-end optimization


