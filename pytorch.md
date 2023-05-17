# PyTorch Fundamentals
PyTorch is an open-source machine learning library that essentially operates as NumPy with GPU acceleration and autograd support. Its core features are the dynamic computational graph in eager execution and the automatic differentiation.

## Dynamic computational graph
The computational graph tracks the sequence of operations that need to be executed when you run your code. The dynamic aspect is that in PyTorch the graph is created on the fly at runtime instead of defined before execution like in TensorFlow. this makes it easier to run, test, and debug code using normal Python. This is also known as eager execution - operations can return concrete values as a result instead of graph operations.

### Graph optimizations
PyTorch models and scripts can be optimized in various ways using TorchScript/JIT compiling.
- Subexpression elimination: removes duplicate operations
- Constant folding: calculating operations with only constants ahead of time
- Dead code elimination: removing any code that does not affect the output
- Operator fusion: multiple operators are combine into a single more efficient operation
- Kernel optimization: use NVIDIA cuDNN or Intel MKL to run common low-level deep learning operations more efficiently

## Autograd
Gradients are tracked in the Tensor class as an attribute. As operations on tensors are coded, the computational graph is created in the background. During backward pass, the graph is traversed backwards and the gradients are calculated and stored in the attribute. The optimizer will then use this to calculate the parameter updates during the optimizer step.

## Training loop
The basic PyTorch training loop consists of:
- Get one batch from the dataset
- Zero out gradients
- Forward pass
- Calculate loss
- Backward pass
- Optimizer step
- Logging

```
for i, batch in enumerate(dataset):
    inputs, labels = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backwards()
    optimizer.step()
    if i % log_every_n_steps == 0:
        log(i, loss)
```

## PyTorch Lightning
Lightning is a wrapper around PyTorch that provides the basic training boilerplate code so users only need to implement the model-specific aspects: forward, training_step, configure_optimizers, validation_step. It does this with the `LightningModule` class, which is essentially `nn.Module` with added functionality.

(GPT) To use PyTorch Lightning, you typically follow these steps:
- Define a LightningModule where you specify your model, the forward method, the loss function, and the optimizer. Also specify what happens in the training, validation, and test steps.
- Optionally, define a LightningDataModule to encapsulate your datasets and data loaders.
- Instantiate the Trainer with any specific options you want (e.g., number of GPUs, whether to use 16-bit precision, etc.)
- Call the fit method of the Trainer, passing in your LightningModule (and LightningDataModule if you defined one).

```
from pytorch_lightning import LightningModule, Trainer

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
trainer = Trainer(max_epochs=10, gpus=1)
```

