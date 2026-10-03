# Next steps
- [x] Implement fully functionnal forward (for conv4 no fully connected Layers)
- [x] Implement dataset parser
- [x] Start multithreading with performance benchmark
- [ ] Create generic Layer Type with Dense and Conv subtype
- [ ] Create Fully Connected Layers
- [ ] Create first untrained network
- [ ] Implement backpropagation (just some conv4 with flipped kernels)
- [ ] Create datasetshuffler
- [ ] Make batch gradient descent
- [ ] Add ADAM Optimiser


# File architecture

```text
includes/
├── debug.h
├── dataset.h
├── network.h
├── layers/
│   ├── layer.h
│   ├── conv_layer.h
│   └──  dense_layer.h
└── tensor/
    ├── tensor.h
    ├── ops_conv.h
    └── ops_lalgebra.h
```