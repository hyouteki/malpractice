#ifndef MALPRACTICE_H_
#define MALPRACTICE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define sigmoid(x) (1.0f/(1.0f+expf(-x)))
#define sigmoid_derivative(x) (x*(1-x))

typedef struct fvec {
	float *vals;
	size_t size;
} fvec;

typedef struct Parameters {
	float learning_rate;
	size_t epochs;
	int log_train_metrics;
} Parameters;

typedef struct Data {
	float *samples;
	size_t *labels;
	size_t num_samples;
	size_t sample_size;
} Data;

typedef struct Model {
	size_t input_size;
	size_t hidden_size;
	size_t output_size;
	
	fvec *input_hidden_weights;
	fvec *hidden_output_weights;
} Model;

fvec *zero_initialize_fvec(size_t size);
fvec *rand_initialize_fvec(size_t size);
void deinitialize_fvec(fvec *vec);
Data *zero_initialize_data(size_t sample_size, size_t num_samples);
void deinitialize_data(Data *data);
Model *initialize_model(size_t input_size, size_t hidden_size, size_t output_size);
void deinitialize_model(Model *model);
void forward(fvec *input, fvec *hidden, fvec *output, Model *model);
void backward(fvec *input, fvec *hidden, fvec *output, fvec *target,
			  float learning_rate, Model *model);
size_t predict(fvec *output);
void train(Data *data, Parameters params, Model *model);
void test(Data *data, Model *model);

fvec *zero_initialize_fvec(size_t size) {
	fvec *vec = (fvec *)malloc(sizeof(fvec));
	vec->vals = (float *)malloc(size*sizeof(float));
	vec->size = size;
	for (size_t i = 0; i < size; ++i) {
		vec->vals[i] = 0.0f;
	}
	return vec;
}

fvec *rand_initialize_fvec(size_t size) {
	fvec *vec = zero_initialize_fvec(size);
	for (size_t i = 0; i < size; ++i) {
		// Random value in range of [-1, 1]
		vec->vals[i] = ((float)rand()/RAND_MAX)*2-1;
	}
	return vec;
}

void deinitialize_fvec(fvec *vec) {
	free(vec);
}

Data *zero_initialize_data(size_t sample_size, size_t num_samples) {
	Data *data = (Data *)malloc(sizeof(data));
	data->samples = (float *)malloc(sizeof(float)*sample_size*num_samples);
	for (size_t i = 0; i < sample_size*num_samples; ++i) {
		data->samples[i] = 0.0f;
	}
	data->labels = (size_t *)malloc(sizeof(size_t)*num_samples);
	for (size_t i = 0; i < num_samples; ++i) {
		data->labels[i] = 0;
	}
	data->num_samples = num_samples;
	data->sample_size = sample_size;
	return data;
}

void deinitialize_data(Data *data) {
	free(data->samples);
	free(data->labels);
	free(data);
}

Model *initialize_model(size_t input_size, size_t hidden_size, size_t output_size) {
	Model *model = (Model *)malloc(sizeof(Model));
	model->input_size = input_size;
	model->hidden_size = hidden_size;
	model->output_size = output_size;
	model->input_hidden_weights = rand_initialize_fvec(input_size*hidden_size);
    model->hidden_output_weights = rand_initialize_fvec(hidden_size*output_size);
	return model;
}

void deinitialize_model(Model *model) {
	deinitialize_fvec(model->input_hidden_weights);
	deinitialize_fvec(model->hidden_output_weights);
	free(model);
}

void forward(fvec *input, fvec *hidden, fvec *output, Model *model) {
	// Input -> Hidden layer
	for (size_t i = 0; i < hidden->size; ++i) {
		hidden->vals[i] = 0;
		for (size_t j = 0; j < input->size; ++j) {
			hidden->vals[i] += input->vals[j]
				*model->input_hidden_weights->vals[i*input->size+j];
		}
		hidden->vals[i] = sigmoid(hidden->vals[i]);
	}

	// Hidden -> Output layer
	for (size_t i = 0; i < output->size; ++i) {
		output->vals[i] = 0;
		for (size_t j = 0; j < hidden->size; ++j) {
			output->vals[i] += hidden->vals[j]
				*model->hidden_output_weights->vals[i*hidden->size+j];
		}
		output->vals[i] = sigmoid(output->vals[i]);
	}
}

// Backward error propagation (gradiant calculation and weight update)
void backward(fvec *input, fvec *hidden, fvec *output, fvec *target,
			  float learning_rate, Model *model) {
	
	float output_errors[output->size];
	float hidden_errors[hidden->size];

	// Calcluate output_errors
	assert(target->size == output->size && "Target vector size /= Output vector size");
	for (size_t i = 0; i < output->size; ++i) {
		output_errors[i] = (target->vals[i]-output->vals[i])
			*sigmoid_derivative(output->vals[i]);
	}

	// Update hidden_output_weights
    for (size_t i = 0; i < output->size; ++i) {
        for (size_t j = 0; j < hidden->size; ++j) {
            model->hidden_output_weights->vals[i*hidden->size+j] +=
				learning_rate*output_errors[i]*hidden->vals[j];
        }
    }

	// Calculate hidden_errors
    for (size_t i = 0; i < hidden->size; ++i) {
        hidden_errors[i] = 0;
        for (size_t j = 0; j < output->size; ++j) {
            hidden_errors[i] += output_errors[j]
				*model->hidden_output_weights->vals[j*hidden->size+i];
        }
        hidden_errors[i] *= sigmoid_derivative(hidden->vals[i]);
    }

	// Update input_hidden_weights
    for (size_t i = 0; i < hidden->size; ++i) {
        for (size_t j = 0; j < input->size; ++j) {
            model->input_hidden_weights->vals[i*input->size+j] +=
				learning_rate*hidden_errors[i]*input->vals[j];
        }
    }
}

size_t predict(fvec *output) {
	size_t max_index = 0;
	for (size_t i = 0; i < output->size; ++i) {
		if (output->vals[i] > output->vals[max_index]) {
			max_index = i;
		}
	}
	return max_index;
}

void train(Data *data, Parameters params, Model *model) {
	fvec *input = zero_initialize_fvec(model->input_size);
	fvec *hidden = zero_initialize_fvec(model->hidden_size);
	fvec *output = zero_initialize_fvec(model->output_size);
	fvec *target = zero_initialize_fvec(model->output_size);

	for (size_t epoch = 0; epoch < params.epochs; ++epoch) {
		int correct = 0;
		for (int sample = 0; sample < data->num_samples; ++sample) {
			// Setup input and target for this sample
			input->vals = &data->samples[sample*input->size];
			for (size_t i = 0; i < target->size; ++i) {
				target->vals[i] = (data->labels[sample] == i)? 1.0f: 0.0f;
			}

			forward(input, hidden, output, model);
			backward(input, hidden, output, target, params.learning_rate, model);

			if (params.log_train_metrics && predict(output) == data->labels[sample]) {
				correct++;
			}
		}
		
		if (params.log_train_metrics) {
			printf("Epoch: %ld, Accuracy: %.2f%%\n",
				   epoch+1, (float)(correct/data->num_samples)*100);
		}
	}
}

void test(Data *data, Model *model) {
	fvec *input = zero_initialize_fvec(model->input_size);
	fvec *hidden = zero_initialize_fvec(model->hidden_size);
	fvec *output = zero_initialize_fvec(model->output_size);

	size_t correct = 0;
	for (size_t sample = 0; sample < data->num_samples; ++sample) {
        input->vals = &data->samples[sample*input->size];
        forward(input, hidden, output, model);

		if (predict(output) == data->labels[sample]) {
            correct++;
        }
    }
	
    printf("Test Accuracy: %.2f%%\n", (float)(correct/data->num_samples)*100);
}

#endif // MALPRACTICE_H_
