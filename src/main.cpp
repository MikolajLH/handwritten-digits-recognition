#include "idx_file.h"
#include "neural_network.h"


int main()
{	
	// MNIST database specific constants and Typedefs for training
	static constexpr size_t image_side = 28;
	static constexpr size_t image_size = image_side * image_side;
	static constexpr size_t number_of_digits = 10;

	using InputVector = mv::ColVector<double, image_size>;
	using OutputVector = mv::ColVector<double, number_of_digits>;

	using DataSet = std::vector<std::pair<InputVector, OutputVector>>;

	static constexpr size_t training_set_size = 50'000;
	static constexpr size_t validation_set_size = 10'000;
	static constexpr size_t test_set_size = 10'000;

	DataSet training_set{};
	DataSet validation_set{};
	DataSet test_set{};


	//loading MNIST files and creating DataSets
	{
		std::vector images_10k = std::move(IDX_File<std::uint8_t>("MNIST_database/t10k-images.idx3-ubyte").data);
		std::vector labels_10k = std::move(IDX_File<std::uint8_t>("MNIST_database/t10k-labels.idx1-ubyte").data);

		std::vector images_60k = std::move(IDX_File<std::uint8_t>("MNIST_database/train-images.idx3-ubyte").data);
		std::vector labels_60k = std::move(IDX_File<std::uint8_t>("MNIST_database/train-labels.idx1-ubyte").data);

		const auto create_dataset = [](const auto& images, const auto& labels, size_t begin_index, size_t size) -> DataSet {
			return std::views::zip_transform(
				[](auto img_v, std::uint8_t lbl) {
					return std::make_tuple(
						InputVector(img_v | std::ranges::to<std::vector>()),
						OutputVector([=](size_t r, size_t c) { return r == lbl; }));
				},
				images |
				std::views::transform([](std::uint8_t x) { return double(x) / 255.; }) |
				std::views::chunk(image_size),
				labels) |
				std::views::drop(begin_index) |
				std::views::take(size) |
				std::ranges::to<DataSet>();
			};

		static_assert(training_set_size + validation_set_size <= 60'000);
		static_assert(test_set_size <= 10'000);
		training_set = create_dataset(images_60k, labels_60k, 0, training_set_size);
		validation_set = create_dataset(images_60k, labels_60k, training_set_size, validation_set_size);

		test_set = create_dataset(images_10k, labels_10k, 0, test_set_size);
	}

	// Creating MLP
	nn::MLP<double, image_size, 500, number_of_digits> mlp(mv::FillTag(1.));
	mlp.randomize();
	mlp.get_function<1>() = mv::MatrixFunction::relu();
	mlp.get_function<2>() = mv::MatrixFunction::softmax();
	//

	const double test_accuracy_init = nn::accuracy(mlp, test_set);
	std::cout << std::format("\n          Test accuracy before training: {:.2f}%", 100. * test_accuracy_init) << "\n\n";


	//launching SGD
	nn::SGD<nn::CostFunction::CrossEntropy, double, image_size, number_of_digits, 500>(
		mlp, 100,
		training_set, validation_set,
		0.01, 5, 10, "nn_H500_B100_MB5.nn"
	);

	const double test_accuracy = nn::accuracy(mlp, test_set);
	std::cout << std::format("\n          Test accuracy after training: {:.2f}%", 100. * test_accuracy) << "\n\n";
	
	return 0;
}

/*
* using Image = mv::Matrix<double, image_side, image_side>;
* const auto show_img = [](const Image& img){

		const auto float_to_char = [](float f) {
			if (f <= 0.30f)return char(219);
			if (f <= 0.50f)return char(178);
			if (f <= 0.70f)return char(177);
			if (f <= 0.90f)return char(176);
			return ' ';};

		for (size_t r = 0; r < img.rows(); ++r) {
			for (size_t c = 0; c < img.cols(); ++c)
				std::cout << float_to_char(1.f - static_cast<float>(img(r, c)));
			std::cout << '\n';
			}
		};
*/