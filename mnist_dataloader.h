#include <stdint.h>
#include <malpractice.h>

Data *mnist_dataloader(const char *image_file_path, const char *label_file_path);

Data *mnist_dataloader(const char *image_file_path, const char *label_file_path) {
	FILE *file = fopen(image_file_path, "rb");
	lodge_assert(file, "Cannot open file");

	uint32_t magic_number;
	fread(&magic_number, sizeof(uint32_t), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	if (magic_number != 2051) {
		fclose(file);
		lodge_fatal("Invalid magic number for mnist images");
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

	file = fopen(label_file_path, "rb");
	lodge_assert(file, "Cannot open file");

	fread(&magic_number, sizeof(uint32_t), 1, file);
	magic_number = __builtin_bswap32(magic_number);
	if (magic_number != 2049) {
		fclose(file);
		lodge_fatal("Invalid magic number for mnist labels");
	}

	uint32_t num_labels;
	fread(&num_labels, sizeof(uint32_t), 1, file);
    num_labels = __builtin_bswap32(num_labels);
	lodge_assert(data->num_samples == num_labels, "#Images /= #Labels");

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
