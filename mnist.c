#include <stdint.h>
#include "malpractice.h"

#define ImagesFilePath "mnist/train-images-idx3-ubyte"
#define LabelsFilePath "mnist/train-labels-idx1-ubyte"
#define TrainSplitPercentage 0.8

Data *data_extraction() {
	FILE *file = fopen(ImagesFilePath, "rb");
	mal_assert(file, "Cannot open file");

	uint32_t magic_number;
	fread(&magic_number, sizeof(uint32_t), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	if (magic_number != 2051) {
		fprintf(stderr, "Invalid magic number for mnist images");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	Data *data = (Data *)malloc(sizeof(Data));

	fread(&data->num_samples, sizeof(uint32_t), 1, file);
	data->num_samples = __builtin_bswap32(data->num_samples);

	uint32_t num_rows, num_cols;
    fread(&num_rows, sizeof(uint32_t), 1, file);
    fread(&num_cols, sizeof(uint32_t), 1, file);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
	data->sample_size = num_rows*num_cols;

	data->samples = (float *)malloc(sizeof(float)*data->num_samples*data->sample_size);
	uint8_t *buffer = (uint8_t *)malloc(data->sample_size);

	for (size_t i = 0; i < data->num_samples; ++i) {
		fread(buffer, sizeof(uint8_t), data->sample_size, file);
		for (size_t j = 0; j < data->sample_size; ++j) {
			data->samples[i*data->sample_size+j] = (float)buffer[j];
		}
	}

	free(buffer);
	fclose(file);

	file = fopen(LabelsFilePath, "rb");
	mal_assert(file, "Cannot open file");

	magic_number;
	fread(&magic_number, sizeof(uint32_t), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	if (magic_number != 2049) {
		fprintf(stderr, "Invalid magic number for mnist labels");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	uint32_t num_labels;
	fread(&num_labels, sizeof(uint32_t), 1, file);
    num_labels = __builtin_bswap32(num_labels);
	mal_assert(data->num_samples == num_labels, "#Images /= #Labels");

	uint8_t *label_buffer = (uint8_t *)malloc(num_labels*sizeof(uint8_t));
    fread(label_buffer, sizeof(uint8_t), num_labels, file);
	data->labels = (size_t *)malloc(num_labels*sizeof(size_t));
	for (size_t i = 0; i < num_labels; ++i) {
		data->labels[i] = (size_t)label_buffer[i];
	}

	free(label_buffer);
	fclose(file);

	return data;
}

void mnist() {
	srand(time(NULL));

	size_t input_size = 784;
	size_t hidden_size = 128;
	size_t output_size = 10;
	
	Model *model = initialize_model(input_size, hidden_size, output_size, Model_Init_Random);
	describe_model(model);

	Data *data = data_extraction();
	describe_data(data);
	normalize_data(data);

	Data *train_data, *test_data;
	partition_data(data, &train_data, &test_data, TrainSplitPercentage);
	deinitialize_data(data);
	
	Parameters params = (Parameters){
		.learning_rate = 0.01f,
		.epochs = 5,
		.log_train_metrics = 1,
	};

    train(train_data, params, model);
    test(test_data, model);
	save_model(model, "mnist/checkpoint1");
	
	deinitialize_model(model);
	deinitialize_data(train_data);
	deinitialize_data(test_data);
}

int main() {
	mnist();
    return 0;
}
