#include <ranges>
#include <cassert>
#include <iterator>
#include <iostream>
#include "neural_network.h"
#include "idx_file.h"
#include <fstream>
#include <filesystem>


void print(const auto& m) {
	for (int i = 0; i < m.rows(); ++i) {
		for (int j = 0; j < m.cols(); ++j) {
			std::cout << m(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

char float_to_char(float f)
{
	if (f <= 0.30f)return char(219);
	if (f <= 0.50f)return char(178);
	if (f <= 0.70f)return char(177);
	if (f <= 0.90f)return char(176);
	return ' ';
}

void print_img(const auto& img) {
	for (size_t r = 0; r < img.rows(); ++r) {
		for (size_t c = 0; c < img.cols(); ++c)
			std::cout << float_to_char(1.f - static_cast<float>(img(r, c)));
		std::cout << '\n';
	}
}


void save(std::string_view path, const std::vector<std::byte>& bytes)
{
	std::ofstream file(std::filesystem::path(path), std::ios::binary);
	file.write((char*)bytes.data(), bytes.size());
}

std::vector<std::byte> load(std::string_view path)
{
	std::ifstream file(std::filesystem::path(path), std::ios::binary);
	file.seekg(0, file.end);
	size_t byte_count = file.tellg();
	file.seekg(0, file.beg);

	std::vector<std::byte> bytes(byte_count);
	file.read((char*)bytes.data(), byte_count);

	return bytes;
}



int main()
{

	nn::MLP<double, 28 * 28, 15, 10> mlp(mv::FillTag(1.));
	mlp.randomize();
	mlp.get_function<1>() = mv::MatrixFunction::relu();
	mlp.get_function<2>() = mv::MatrixFunction::softmax();

	auto images_file = IDX_File("MNIST_database/t10k-images.idx3-ubyte");
	auto labels_file = IDX_File("MNIST_database/t10k-labels.idx1-ubyte");

	const size_t N = 10'000;
	auto images = std::move(images_file.data_uint8_t);
	auto labels = std::move(labels_file.data_uint8_t);

	std::vector<std::pair<mv::Matrix<double, 28 * 28, 1>, mv::Matrix<double, 10, 1>>> training_data{};
	training_data.reserve(N);

	for (size_t i = 0; i < N; ++i) {
		training_data.emplace_back(
			[&](size_t r, size_t c) {
				return double(images[i * 28 * 28 + r]) / 255.;
			},
			[&](size_t r, size_t c) {
				return double(labels[i] == r);
			}
		);
	}

	const size_t batch_size = 100;
	const size_t batches = N / batch_size;
	const size_t epochs = 1000;
	double eta = 0.001;

	std::random_device rd;
	std::mt19937 g(rd());


	for (size_t e = 0; e < epochs; ++e) {
		std::shuffle(training_data.begin(), training_data.end(), g);
		mlp.reset_errors();
		double epoch_error = 0.;
		for (const auto& batch : training_data | std::views::chunk(batch_size)) {
			double batch_error = 0.;
			mlp.reset_errors();
			for (const auto& [X, Y] : batch) {
				mlp.get_input() = X;
				mlp.feed_forward();
				const auto A = mlp.get_output();
				const auto dCx = A - Y;
				const auto Cx = 0.5 * mv::mat_mul(dCx.transpose(), dCx)(0, 0);
				batch_error += Cx;
				mlp.backpropagate(dCx);
			}
			batch_error /= double(batch_size);
			mlp.multiply_weights_deltas((eta) / (double(batch_size)));
			mlp.apply_weights_deltas();
			epoch_error += batch_error;
		}
		epoch_error /= double(batches);
		std::cout << "Epoch: " << e << " Epoch error: " << epoch_error << "\n";

		save("w0.mv", mv::serialize(mlp.get_weights<0>()));
		save("w1.mv", mv::serialize(mlp.get_weights<1>()));
	}

	for (size_t i = 0; i < 10 * 28 * 28; i += 28 * 28) {
		size_t image_begin = 28 * 28 * i;
		const auto X = mv::Matrix<double, 28 * 28, 1>(
			[&](size_t r, size_t c) {
				return images[image_begin + r];
			});
		mlp.get_input() = X;
		mlp.feed_forward();
		const auto Y = mlp.get_output();

		const auto img = X.reshape<28, 28>();
		print_img(img);
		std::cout << "|";
		for (size_t digit = 0; auto & v : Y.transpose().data)
			std::cout << (digit++) << ": " << int(v * 100.) << "%  |";
		std::cout << "\n\n";
	}

	return 0;
}