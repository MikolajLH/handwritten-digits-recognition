#include "idx_file.h"
#include <fstream>
#include <filesystem>
#include <cstddef>
#include <array>
#include <iostream>
#include <bit>
#include <algorithm>
#include <functional>
#include <numeric>
#include <ranges>


IDX_File::IDX_File(std::string_view path) {
	std::cout << "path: " << path << '\n';

	std::ifstream file(std::filesystem::path(path), std::ios::binary);
	if (not file.good())throw std::logic_error("failed to load file");

	auto file_iterator = std::istreambuf_iterator<char>(file);
	auto file_end = std::istreambuf_iterator<char>{};

	std::array<std::byte, 4u>magic_number{};
	for (auto& byte : magic_number) {
		byte = std::byte(*file_iterator);
		++file_iterator;
		if (file_iterator == file_end)throw std::logic_error("file ended too soon, couldn't load magic_number");
	}
	if (not (magic_number[0] == std::byte(0) and magic_number[1] == std::byte(0)))throw std::logic_error("first to bytes of magic number should always be zero");

	auto data_type = std::to_integer<std::uint8_t>(magic_number[2]);
	auto number_of_dimensions = std::to_integer<std::uint8_t>(magic_number[3]);
	std::string_view data_type_str{};

	switch (data_type) {
		case 0x08:
			data = std::vector<std::uint8_t>{};
			data_type_str = "unsigned byte";
			this->type = "std::uint8_t";
			break;
		case 0x09:
			data = std::vector<std::int8_t>{};
			data_type_str = "signed byte";
			this->type = "std::int8_t";
			break;
		case 0x0B:
			data = std::vector<std::int16_t>{};
			data_type_str = "short (2 bytes)";
			this->type = "std::int16_t";
			break;
		case 0x0C:
			data = std::vector<std::int32_t>{};
			data_type_str = "int (4 bytes)";
			this->type = "std::int32_t";
			break;
		case 0x0D:
			data = std::vector<float>{};
			data_type_str = "float (4 bytes)";
			this->type = "float";
			if constexpr (sizeof(float) != 4) {
				throw std::logic_error("size of float should be 4 bytes");
			}
			break;
		case 0x0E:
			data = std::vector<double>{};
			data_type_str = "double (8 bytes)";
			this->type = "double";
			if constexpr (sizeof(double) != 8) {
				throw std::logic_error("size of double should be 8 bytes");
			}
			break;
	}

	std::cout << "data type: " << std::hex << size_t(data_type) << " -> " << data_type_str << " -> " << this->type << "\n";
	std::cout << "number of dimensions: " << std::dec << size_t(number_of_dimensions) << "\n";



	std::vector<std::uint32_t>dimensions(number_of_dimensions);
	std::array<std::byte, 4u>buffer{};
	for (size_t i = 0u; auto & d : dimensions) {
		for (auto& byte : buffer) {
			byte = std::byte(*file_iterator);
			++file_iterator;
			if (file_iterator == file_end)throw std::logic_error("file ended too soon, couldn't load sizes in dimensions");
		}
		d = std::bit_cast<std::uint32_t>(buffer);
		if constexpr (std::endian::native != std::endian::big) {
			d = std::byteswap(d);
		}
		std::cout << "dim " << (++i) << ": " << d << "\n";
	}
	
	auto data_size = std::accumulate(std::begin(dimensions), std::end(dimensions), size_t(1), std::multiplies<size_t>());
	std::visit(
		[=]
		(auto& vec) {
			vec.resize(data_size); },
			data);

	
	std::cout << "data_size: " << data_size << "\n";
	auto vector_loader = [&]<typename T>(std::vector<T>&vec) {
		static constexpr size_t word_size = sizeof(T);
		std::array<std::byte, word_size>buffer{};
		size_t chunks_counter = 0u;
		for (T& e : vec) {
			for (auto& byte : buffer) {
				if (file_iterator == file_end)throw std::logic_error("file ended too soon, couldn't load chunk of data");
				byte = std::byte(*file_iterator);
				++file_iterator;
			}

			if constexpr (word_size > 1u and std::endian::native != std::endian::big) {
				auto temp = buffer;
				for (auto [a, b] : std::views::zip(buffer, std::views::reverse(buffer))) {
					a = b;
				}
			}
			e = std::bit_cast<T>(buffer);
			++chunks_counter;
		}

		if (file_iterator == file_end) {
			std::cout << "File ended as expected, loaded " << chunks_counter << " chunks of data\n";
		}
		else {
			throw std::logic_error("file was expected to end here but it didn't");
		}
	};
	std::visit(vector_loader, data);

	std::visit([this]<typename T>(std::vector<T>& vec) {
		if constexpr (std::same_as<T, std::uint8_t>) {
			this->data_uint8_t = std::move(vec);
		}
		if constexpr (std::same_as<T, std::int8_t>) {
			this->data_int8_t = std::move(vec);
		}
		if constexpr (std::same_as<T, std::int16_t>) {
			this->data_int16_t = std::move(vec);
		}
		if constexpr (std::same_as<T, std::int32_t>) {
			this->data_int32_t = std::move(vec);

		}
		if constexpr (std::same_as<T, float>) {
			this->data_float = std::move(vec);
		}
		if constexpr (std::same_as<T, double>) {
			this->data_double = std::move(vec);
		}
	}, data);
	std::cout << "\n";
}