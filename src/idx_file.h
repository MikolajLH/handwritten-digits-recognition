#pragma once
#include <string_view>
#include <cstdint>
#include <vector>
#include <variant>
#include <stdexcept>

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


class IDX_File
{
	public:
		IDX_File(std::string_view path) ;

		std::string_view type{};
		std::vector<std::uint8_t> data_uint8_t{};
		std::vector<std::int8_t> data_int8_t{};
		std::vector<std::int16_t> data_int16_t{};
		std::vector<std::int32_t> data_int32_t{};
		std::vector<float> data_float{};
		std::vector<double> data_double{};

	private:
		std::variant<
			std::vector<std::uint8_t>,
			std::vector<std::int8_t>,
			std::vector<std::int16_t>,
			std::vector<std::int32_t>,
			std::vector<float>,
			std::vector<double>
		> data{};
};