#pragma once
#include <iostream>
#include <string_view>
#include <cstdint>
#include <vector>
#include <variant>
#include <numeric>
#include <stdexcept>
#include <array>
#include <bit>
#include <fstream>
#include <filesystem>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <ranges>

/*
* Description taken from http://yann.lecun.com/exdb/mnist/
* 
* 

THE IDX FILE FORMAT
the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.

The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest. 
*/

template<class T>
class IDX_File
{
	public:
		IDX_File(std::string_view path, bool verbose = true) {
			if (verbose)
				std::cout << "path: " << path << '\n';

			//load file
			std::ifstream file(std::filesystem::path(path), std::ios::binary);
			if (not file.good())
				throw std::logic_error("failed to load file");

			auto file_iterator = std::istreambuf_iterator<char>(file);
			auto file_end = std::istreambuf_iterator<char>{};


			//load magic number
			std::array<std::byte, 4u>magic_number{};
			for (auto& byte : magic_number) {
				byte = std::byte(*file_iterator);
				++file_iterator;
				if (file_iterator == file_end)
					throw std::logic_error("file ended too soon, couldn't load magic_number");
			}
			if (not (magic_number[0] == std::byte(0) and magic_number[1] == std::byte(0)))
				throw std::logic_error("first 2 bytes of magic number should always be zero");

			//interpret magic number
			const auto data_type_code = std::to_integer<std::uint8_t>(magic_number[2]);
			const auto number_of_dimensions = std::to_integer<std::uint8_t>(magic_number[3]);
			std::string_view data_type_code_str{};

			switch (data_type_code) {
			case 0x08:
				data_type_code_str = "unsigned byte";
				data_type = std::uint8_t{};
				data_type_str = "std::uint8_t";
				break;
			case 0x09:
				data_type_code_str = "signed byte";
				data_type = std::int8_t{};
				data_type_str = "std::int8_t";
				break;
			case 0x0B:
				data_type_code_str = "short (2 bytes)";
				data_type = std::int16_t{};
				data_type_str = "std::int16_t";
				break;
			case 0x0C:
				data_type_code_str = "int (4 bytes)";
				data_type = std::int32_t{};
				data_type_str = "std::int32_t";
				break;
			case 0x0D:
				data_type_code_str = "float (4 bytes)";
				data_type = float{};
				data_type_str = "float";
				if constexpr (sizeof(float) != 4) {
					throw std::logic_error("size of float should be 4 bytes");
				}
				break;
			case 0x0E:
				data_type_code_str = "double (8 bytes)";
				data_type = double{};
				data_type_str = "double";
				if constexpr (sizeof(double) != 8) {
					throw std::logic_error("size of double should be 8 bytes");
				}
				break;
			default:
				throw std::logic_error("unexpected type");
			}

			if (verbose) {
				std::cout << "data type: " << std::hex << size_t(data_type_code) << " -> " << data_type_code_str << " -> " << data_type_str << "\n";
				std::cout << "number of dimensions: " << std::dec << size_t(number_of_dimensions) << "\n";
			}

			// check whether provided template type is correct 
			bool good_type = false;
			std::visit([&]<typename K>(K) {
				if constexpr (std::same_as<T, K>) {
					good_type = true;
				}
			}, data_type);

			if (not good_type)
				throw std::logic_error("Type provided in template argument is inconsistent with type of data inside the file");

			// load dimensions
			std::vector<std::uint32_t>dimensions(number_of_dimensions);
			{
				std::array<std::byte, 4u>buffer{};
				for (size_t i = 0u; auto & d : dimensions) {
					for (auto& byte : buffer) {
						byte = std::byte(*file_iterator);
						++file_iterator;
						if (file_iterator == file_end)
							throw std::logic_error("file ended too soon, couldn't load sizes in dimensions");
					}
					d = std::bit_cast<std::uint32_t>(buffer);
					if constexpr (std::endian::native != std::endian::big) {
						d = std::byteswap(d);
					}
					if (verbose)
						std::cout << "dim " << (++i) << ": " << d << "\n";
				}
			}
			const auto data_size = std::accumulate(std::begin(dimensions), std::end(dimensions), size_t(1), std::multiplies<size_t>());
			if (verbose)
				std::cout << "data_size: " << data_size << "\n";

			data.resize(data_size);
			static constexpr size_t data_type_size = sizeof(T);

			{
				std::array<std::byte, data_type_size>buffer{};
				size_t components_counter = 0u;
				for (T& e : data) {
					for (auto& byte : buffer) {
						if (file_iterator == file_end)throw std::logic_error("file ended too soon, couldn't load alll components");
						byte = std::byte(*file_iterator);
						++file_iterator;
					}

					e = std::bit_cast<T>(buffer);
					if constexpr (data_type_size > 1u and std::endian::native != std::endian::big) {
						e = std::byteswap(e);
					}
					++components_counter;
				}
				if (file_iterator != file_end)
					throw std::logic_error("file was expected to end here but it didn't");
				else if (verbose)
					std::cout << "File ended as expected, loaded " << components_counter << " chunks of data\n\n";
			}
			
		}

		std::vector<T>data{};
		std::string_view data_type_str{};

	private:
		std::variant<
			std::uint8_t,
			std::int8_t,
			std::int16_t,
			std::int32_t,
			float,
			double> data_type{};
};