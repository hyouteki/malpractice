#include "malpractice.h"

int main() {
    srand(time(NULL));

	size_t input_size = 784;
	size_t hidden_size = 128;
	size_t output_size = 10;
	
	Model *model = initialize_model(input_size, hidden_size, output_size);

	size_t num_samples = 1000;
	Data *data = zero_initialize_data(input_size, num_samples);

	Parameters params = (Parameters){
		.learning_rate = 0.01f,
		.epochs = 5,
		.log_train_metrics = 1,
	};
	
    train(data, params, model);
    test(data, model);

	deinitialize_model(model);
	deinitialize_data(data);
    return 0;
}
