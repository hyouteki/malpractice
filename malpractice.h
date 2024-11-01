#ifndef MALPRACTICE_H_
#define MALPRACTICE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define sigmoid(x) (1.0f/(1.0f+expf(-x)))
#define sigmoid_derivative(x) (x*(1-x))
#define mal_assert(condition, message)							\
    do {														\
        if (!(condition)) {										\
            fprintf(stderr, "Assertion Failed: %s\n", message);	\
            exit(EXIT_FAILURE);									\
        }														\
    } while (0)

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

typedef enum {
	Model_Init_Zeroes,
	Model_Init_Random,
	Model_Init_Xavier,
} Model_InitTechnique;

typedef struct Model {
    size_t input_size;
    size_t hidden_size;
    size_t output_size;

	Model_InitTechnique tech;
    fvec *input_hidden_weights;
    fvec *hidden_output_weights;
} Model;

fvec *zero_initialize_fvec(size_t size);
fvec *rand_initialize_fvec(size_t size);
fvec *xavier_initialize_fvec(size_t n_in, size_t n_out);
void set_uniform_fvec(fvec *vec, float limit);
fvec *read_fvec_from_file(FILE *file);
void write_fvec_to_file(fvec *vec, FILE *file);
void deinitialize_fvec(fvec *vec);

Data *zero_initialize_data(size_t sample_size, size_t num_samples);
void describe_data(Data *data);
void partition_data(Data *data, Data **split1, Data **split2, float fraction);
void shuffle_data(Data *data);
void normalize_data(Data *data);
void deinitialize_data(Data *data);

const char *model_init_technique_name(Model_InitTechnique tech);
Model *initialize_model(size_t input_size, size_t hidden_size, size_t output_size,
						Model_InitTechnique tech);
void describe_model(Model *model);
Model *load_model(const char *filepath);
void save_model(Model *model, const char *filepath);
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
    set_uniform_fvec(vec, 1);
    return vec;
}

fvec *xavier_initialize_fvec(size_t n_in, size_t n_out) {
    fvec *vec = zero_initialize_fvec(n_in*n_out);
    float limit = sqrt(6.0f/(n_in+n_out));
    set_uniform_fvec(vec, limit);
	return vec;
}

void set_uniform_fvec(fvec *vec, float limit) {
	for (size_t i = 0; i < vec->size; ++i) {
		// Uniform distribution between [-limit, limit]
        vec->vals[i] = ((float)rand()/RAND_MAX)*2*limit - limit;
    }
}

fvec *read_fvec_from_file(FILE *file) {
    fvec *vec = (fvec *)malloc(sizeof(fvec));
    fread(&vec->size, sizeof(size_t), 1, file);
    vec->vals = (float *)malloc(sizeof(float)*vec->size);
    fread(vec->vals, sizeof(float), vec->size, file);
    return vec;
}

void write_fvec_to_file(fvec *vec, FILE *file) {
	fwrite(&vec->size, sizeof(size_t), 1, file);
	fwrite(vec->vals, sizeof(float), vec->size, file);
}

void deinitialize_fvec(fvec *vec) {
    free(vec);
}

Data *zero_initialize_data(size_t sample_size, size_t num_samples) {
    Data *data = (Data *)malloc(sizeof(Data));
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

void describe_data(Data *data) {
	printf("Sample-Size: %ld, #Samples: %ld\n", data->sample_size, data->num_samples);
}

// NOTE: splits should not be initialized before the calling of this function,
//       the original data is not deinitialized
void partition_data(Data *data, Data **split1, Data **split2, float fraction) {
	mal_assert(fraction >= 0 && fraction < 1, "Fraction should lie in [0, 1]");

	size_t split1_num_samples = (size_t)(data->num_samples*fraction);
	*split1 = zero_initialize_data(data->sample_size, split1_num_samples);
	memcpy((*split1)->samples, data->samples,
		   sizeof(float)*split1_num_samples*data->sample_size);
	memcpy((*split1)->labels, data->labels, sizeof(size_t)*split1_num_samples);

	size_t split2_num_samples = data->num_samples-split1_num_samples;
	*split2 = zero_initialize_data(data->sample_size, split2_num_samples);
	memcpy((*split2)->samples, data->samples+data->sample_size*split1_num_samples,
		   sizeof(float)*split2_num_samples*data->sample_size);
	memcpy((*split2)->labels, data->labels+data->sample_size,
		   sizeof(size_t)*split2_num_samples);
}

void shuffle_data(Data *data) {
	size_t n = data->num_samples, ss = data->sample_size;
	size_t *shuffle_list = (size_t *)malloc(n*sizeof(size_t));
	for (size_t i = 0; i < n; ++i) {
	    shuffle_list[i] = i;
	}
	for (size_t i = 0; i < n; ++i) {
		size_t rand_i = rand()%n;
		size_t tmp = shuffle_list[i];
	    shuffle_list[i] = shuffle_list[rand_i];
		shuffle_list[rand_i] = tmp;
	}

    float *new_samples = (float *)malloc(n*ss*sizeof(float));
	size_t *new_labels = (size_t *)malloc(n*sizeof(size_t));

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < ss; ++j) {
			new_samples[i*ss+j] = data->samples[shuffle_list[i]*ss+j];
		}
		new_labels[i] = data->labels[shuffle_list[i]];
	}

	free(data->samples);
	free(data->labels);
	free(shuffle_list);
	data->samples = new_samples;
	data->labels = new_labels;
}

// Normalizes the data samples to be in the range of [0, 1]
void normalize_data(Data *data) {
	float min_v = data->samples[0], max_v = data->samples[0];
    for (size_t i = 0; i < data->sample_size*data->num_samples; ++i) {
		if (data->samples[i] < min_v) {
			min_v = data->samples[i];
		}
		if (data->samples[i] > max_v) {
			max_v = data->samples[i];
		}
	}
	
    mal_assert(max_v != min_v, "Normalization range cannot be zero");
	for (size_t i = 0; i < data->sample_size*data->num_samples; ++i) {
		data->samples[i] = (data->samples[i]-min_v)/(max_v-min_v);
	}
}

void deinitialize_data(Data *data) {
    free(data->samples);
    free(data->labels);
    free(data);
}

const char *model_init_technique_name(Model_InitTechnique tech) {
	switch (tech) {
	case Model_Init_Zeroes:
		return "Zeroes";	
	case Model_Init_Random:
		return "Random";
	case Model_Init_Xavier:
	    return "Xavier";
	default:
		mal_assert(0, "Invalid model initialization technique");
	}
}

Model *initialize_model(size_t input_size, size_t hidden_size, size_t output_size,
						Model_InitTechnique tech) {
	
    Model *model = (Model *)malloc(sizeof(Model));
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->output_size = output_size;

	model->tech = tech;
	switch (tech) {
	case Model_Init_Zeroes:
		model->input_hidden_weights = zero_initialize_fvec(input_size*hidden_size);
		model->hidden_output_weights = zero_initialize_fvec(hidden_size*output_size);
		break;
	case Model_Init_Random:
		model->input_hidden_weights = rand_initialize_fvec(input_size*hidden_size);
		model->hidden_output_weights = rand_initialize_fvec(hidden_size*output_size);
		break;
	case Model_Init_Xavier:
		model->input_hidden_weights = xavier_initialize_fvec(input_size, hidden_size);
		model->hidden_output_weights = xavier_initialize_fvec(hidden_size, output_size);
		break;
	default:
		mal_assert(0, "Invalid model initialization technique");
	}

    return model;
}

void describe_model(Model *model) {
	printf("Input-Size: %ld, Hidden-Size: %ld, Output-Size: %ld, "
		   "Init-Technique: %s\n", model->input_size,
		   model->hidden_size, model->output_size,
		   model_init_technique_name(model->tech));
}

Model *load_model(const char *filepath) {
	FILE *file = fopen(filepath, "wb");
	mal_assert(file, "Cannot open file to load model");

	Model *model = (Model *)malloc(sizeof(Model));
	fread(&model->input_size, sizeof(size_t), 1, file);
	fread(&model->hidden_size, sizeof(size_t), 1, file);
	fread(&model->output_size, sizeof(size_t), 1, file);
	fread(&model->tech, sizeof(Model_InitTechnique), 1, file);
	model->input_hidden_weights = read_fvec_from_file(file);
	model->hidden_output_weights = read_fvec_from_file(file);

	fclose(file);
	return model;
}

void save_model(Model *model, const char *filepath) {
	FILE *file = fopen(filepath, "wb");
	mal_assert(file, "Cannot open file to save model");

	fwrite(&model->input_size, sizeof(size_t), 1, file);
	fwrite(&model->hidden_size, sizeof(size_t), 1, file);
	fwrite(&model->output_size, sizeof(size_t), 1, file);
	fwrite(&model->tech, sizeof(Model_InitTechnique), 1, file);
	write_fvec_to_file(model->input_hidden_weights, file);
	write_fvec_to_file(model->hidden_output_weights, file);
	fclose(file);

	printf("Model saved at '%s'\n", filepath);
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
    mal_assert(target->size == output->size, "Target vector size /= Output vector size");
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
        size_t correct = 0;
		clock_t start = clock();
        for (size_t sample = 0; sample < data->num_samples; ++sample) {
            // Setup input and target for this sample
            memcpy(input->vals, &data->samples[sample*input->size], data->sample_size);
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
			double time_elapsed = (double)(clock()-start)/CLOCKS_PER_SEC;
            printf("Epoch: %ld, Accuracy: %.2f%% (%ld/%ld), Time: %.2fs (%.2fit/s)\n",
                   epoch+1, ((float)correct/data->num_samples)*100,
				   correct, data->num_samples, time_elapsed,
				   (float)data->num_samples/time_elapsed);
        }
    }
}

void test(Data *data, Model *model) {
    fvec *input = zero_initialize_fvec(model->input_size);
    fvec *hidden = zero_initialize_fvec(model->hidden_size);
    fvec *output = zero_initialize_fvec(model->output_size);

    size_t correct = 0;
    for (size_t sample = 0; sample < data->num_samples; ++sample) {
        memcpy(input->vals, &data->samples[sample*input->size], data->sample_size);
        forward(input, hidden, output, model);

        if (predict(output) == data->labels[sample]) {
            correct++;
        }
    }
    
    printf("Test Accuracy: %.2f%% (%ld/%ld)\n",
		   ((float)correct/data->num_samples)*100, correct, data->num_samples);
}

#endif // MALPRACTICE_H_
