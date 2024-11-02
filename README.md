> A lightweight C library for creating, training, and evaluating simple neural networks with customizable initialization methods.
> This library supports basic forward and backward propagation, sigmoid activation, data handling, and model saving/loading.
---

## Features
1. **Custom Initialization Techniques**: Supports zero, random, and Xavier initialization.
2. **Forward and Backward Propagation**: Implements core functions for feedforward and backpropagation.
3. **Data Handling**: Functions to initialize, normalize, partition, and describe data.
4. **Model Serialization**: Save and load model configurations to/from files.

## Usage
1. **Model Initialization**: Set input, hidden, and output sizes with your preferred initialization method.
2. **Training and Testing**: Use train and test functions with Data and Parameters structures to train the model.
3. **Model Persistence**: Save and load models using save_model and load_model.

## API Overview
### Core Structures
1. **Model**: Represents the neural network with layer sizes and weight initialization.
2. **Data**: Holds data samples and labels for training/testing.
3. **Parameters**: Configurable training parameters.
### Key Functions
1. **Model Initialization**: `initialize_model()`
2. **Forward and Backward Propagation**: `forward()`, `backward()`
3. **Data Manipulation**: `normalize_data()`, `partition_data()`, `describe_data()`
4. **Model Persistence**: `save_model()`, `load_model()`

## Getting Started
```bash
git clone https://github.com/hyouteki/malpractice --recursive --depth=1
cd malpractice
./build.sh
```

## Example
```c
#include "malpractice.h"

int main() {
    // Model parameters
    size_t input_size = 784, hidden_size = 128, output_size = 10;
    Model_InitTechnique init_tech = Model_Init_Xavier;
    Model *model = initialize_model(input_size, hidden_size, output_size, init_tech);

    // Load data (replace with actual loading logic)
    Data *data = zero_initialize_data(input_size, 100);

    // Set training parameters
    Parameters params = {.learning_rate = 0.01, .epochs = 1000, .log_train_metrics = 1};

    // Train and evaluate
    train(data, params, model);
    test(data, model);

    // Save model
    save_model(model, "model.bin");

    // Cleanup
    deinitialize_data(data);
    deinitialize_model(model);
    return 0;
}
```
