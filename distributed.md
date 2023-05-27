# Distributed Training
Distributed training of a model involves sharding the data, model weights, gradients, and/or optimizer state across multiple devices. This leverages multiple devices to accelerate training, but involves communication overhead to maintain synchronization. In PyTorch Distributed, There are two strategies: DDP and FSDP.

## Distributed Data Parallelism
DDP shares the input batch, so each device sees a different subset of the batch. The model parameters are replicated across devices. This enables training with larger batches as they will consume less memory on device. However, larger models that may not fit on one device cannot take advantage of DDP. DDP also adds communication overhead in the all_reduce operation that averages the gradients across different data shards.

## Fully Sharded Data Parallelism
FSDP additionally shards the model and optimizer state. Now, each device has a different shard of the batch, model parameters, and optimizer parameters (first and second order moments in Adam, for example). A forward/backward pass would involve running through the model shards in the correct sequence, and each shard can compute its own gradients accordingly. For cross-shard dependencies to compute gradients, some additional synchronization or all-reduce operations is required.

FSDP is quite useful for larger models that do not fit on one device. However, it introduces significant communication overhead, which increases run time (forward + backward + update). For larger models this tradeoff is usually worth it since you can increase the batch size with the extra memory and make up for increase train time. For smaller models, this may not be worth it if DDP is sufficient enough to fit the model. Finding when the increased batch size outweighs the communication overhead depends on the model, task at hand, and available computational resources and may require experimentation.

Another caveat with FSDP is that batch normalization layers usually require computing global statistics across shards. This will incur another all_reduce operation but the model sharding benefits may still be worthwhile.

## Mixed precision
Training with mixed precision involves training with fp16 data. It is "mixed" because certain operations still require full floating point precision. For example, normalization layers usually requires fp32, so the inputs are autocast to fp32, normalized, then cast back. The advantage of using fp16 is that it takes up half the memory, thus you can increase the batch or model size if needed or better utilize GPUs. The disadvantages are that training may become unstable for models with normalization layers. Sometimes the cast back from fp32 to fp16 can lead to NaNs if the fp32 numbers are too precise or outside the range to be represented in fp16.

An alternative to fp16 is bf16. It takes up the same memory as fp16, but represents the same range of numbers as fp32. It does this by matching the number of exponent bits to fp32 (8 bits) by sacrificing the mantissa/fraction bits (7 bits in bf16 vs 10 bits in fp16). As a result, the same range of numbers can be represented but with less precision. Recall that the fraction bits determine the granularity of the number and the exponent bits determine the range. Usually, this decreased precision is acceptable for machine learning applications, and bf16 enjoys the benefits of fp16 without the drawbacks.

## Activation checkpointing
