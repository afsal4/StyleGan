# StyleGAN v1 From Scratch (PyTorch)

A from-scratch PyTorch implementation of **StyleGAN v1**, trained on Kaggle using 2 GPUs with Distributed Data Parallel (DDP). The model uses progressive growing to generate images at resolutions up to **512×512**.

## Results

<p style="text-align: center;">
  <img src="stylegan.png" width="70%">
</p>

**Training Progress:** [Watch the video](https://chatgpt.com/c/output.mp4)

### FID Score

Achieved an amazing improvement in FID score, from **[STARTING_FID]** down to **0.173**.

## Model Architecture

* **Mapping Network:** maps the input latent vector `z` to an intermediate latent space `w`, which controls style at each resolution via AdaIN.
* **Synthesis Network:** progressively grown generator, starting at 4×4 and growing up to 512×512, with learned constant input, noise injection at every block, and style modulation (AdaIN) driven by `w`.
* **Discriminator:** mirrors the generator's progressive growing structure, with minibatch standard deviation for diversity and a matching fade-in schedule for new layers.
* **Progressive Growing:** both generator and discriminator grow in resolution together, with alpha-blended fade-in transitions between resolution stages to stabilize training.

## Features

* Progressive growing from 4×4 up to 512×512 resolution.
* Style-based generation with a mapping network and noise injection.
* Distributed training using 2 GPUs.
* FID score evaluation during training.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training on a machine with 2 available GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

The model was trained on Kaggle. You can view the notebook and access the model weights and outputs here:

[**Kaggle Notebook**](https://www.kaggle.com/code/afsalali/stylegan)

## Model Weights & Inference

**Pretrained weights are not included.** Training saves `checkpoint.pt` and `save_best.pt` in the working directory.

For inference, load the matching generator configuration and weights, then pass a random latent vector to the generator. For **512×512**, use `step=8`, `psi=0.7`, and `inference_mode=True`.

The generator output is in the `[-1, 1]` range and should be converted to an image format for visualization.
