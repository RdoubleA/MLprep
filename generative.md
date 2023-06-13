# Generative AI

- [Autoencoders](#autoencoders)
- [GANs](#gans)
- [Diffusion models](#diffusion-models)
- [Large language models](#large-language-models)

## Diffusion models
Diffusion models are a class of generative models that learn a "diffusion process" and attempt to reverse it. This diffusion process is a stochastic Markovian process that slowly transforms one statistical distribution to another. Typically, your data is the starting distribution, and the final distribution is Gaussian noise. The forward process involves slowly corrupting the data distribution across many timesteps until it becomes indistinguishable from Gaussian noise, the verse process would be denoising it to create a data sample. Diffusion models can be any architecture that learns to predict the noise level in a data sample at a given diffusion timestep by predicting the "cleaned" version.

### DDPM
Denoising Diffusion Probabilistic Models uses a UNet model to take in an input (noised image from previous timestep) and denoises it by one timestep. It uses a standard MSE loss function. Using the mathematics of the diffusion process and a given data sample, we already know what the noised image would look like at every time step. We train the model to learn to denoise for every timestep in the reverse diffusion step. Training involves randomly sampling a timestep, generating the corrupted image, and forward passing the model. Because we should sample every timestep during training, and we have to run diffusion calculations for each pass, training DDPMs is very slow. During inference, the entire reverse diffusion process must be run to generate a sample, usually 1000 timesteps, making inference slow as well. But the quality of images generated outperforms GANs. It is also easier to train and more scalable since we can leverage UNet or transformer-like architectures.

### DDIM
Denoising Diffusion Implicit Models improve on DDPMs by introducing a sampling technique that allows you to subsample the timesteps during inference, speeding up generation. This is done using annealed Langevin Dynamics. However, this adds complexity to the model training, and may not produce as high resolution images as DDPMs.

### Cascaded diffusion models
Instead of having one model learn the entire reverse diffusion process, you can chain multiple models. Each models will train with fewer timesteps, so individually they may train faster. But you need to coordinate the training of multiple models and it will likely have more total parameters than a single model, so you need enough compute and data to do this.

### Upsampling
Similar to the idea of cascaded diffusion models, except you chain models to increase the resolution of the image. The first model might go from noise to 64x64, then the next model doubles it, and so on. Again you are adding more complexity and parameters, but this could generate much higher resolution images

### Conditioned diffusion models
Diffusion models by default are conditioned with a timestep embedding. This is usually a sinusoidal embedding and a linear projection of the timestep integer. You can add additional conditioning inputs, such as text caption embeddings. These are typically used in cross-attention in the UNet model. Conditioning inputs allow you to control the generation process and create specific samples from noise.

### Stable/latent diffusion
DDPMs typically operate directly in the image space. Stable diffusion works in the latent space. It achieves this by first trraining an autoencoder to map an image to a latent space. Then, the denoising UNet learns to transform from Gaussian noise into a latent representation of the image. To generate the image, the UNet output is fed into the decoder of the autoencoder to decode the latent representation. Typically, stable diffusion models are conditioned with a text caption embedding that is used in the model via cross-attention. By operating in the latent space, stable diffusion reduces complexity and increases generation speed.

### DALLE-2

