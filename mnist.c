#include <stdint.h>
#include <malpractice.h>
#include <mnist_dataloader.h>

#define ImagesFilePath "mnist/train-images-idx3-ubyte"
#define LabelsFilePath "mnist/train-labels-idx1-ubyte"
#define TrainSplitPercentage 0.8

void mnist() {
	srand(time(NULL));

	size_t input_size = 784;
	size_t hidden_size = 128;
	size_t output_size = 10;
	
	Model *model = initialize_model(input_size, hidden_size, output_size, Model_Init_Random);
	describe_model(model);

	Data *data = mnist_dataloader(ImagesFilePath, LabelsFilePath);
	shuffle_data(data);
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
	save_model(model, "mnist_checkpoint");
	
	deinitialize_model(model);
	deinitialize_data(train_data);
	deinitialize_data(test_data);
}

int main() {
	mnist();
    return 0;
}
