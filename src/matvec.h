#pragma once
#include <vector>
#include <iterator>
#include <cassert>
#include <type_traits>
#include <concepts>
#include <cmath>
#include <functional>
#include <span>
#include <random>
#include <initializer_list>
#include <bit>
#include <array>
#include <iostream>

namespace mv
{
	namespace concepts
	{
		template<typename Fn, class X>
		concept matrix_indices_function = std::regular_invocable<Fn, size_t, size_t> and std::convertible_to<std::invoke_result_t<Fn, size_t, size_t>, X>;

		template<typename Fn, typename X>
		concept function = std::regular_invocable<Fn, X> and std::convertible_to<std::invoke_result_t<Fn, X>, X>;
	}

	auto& matrix_at(std::contiguous_iterator auto begin, size_t r, size_t c, size_t R, size_t C) {
		assert((r * C + c < R * C));
		return *(begin + r * C + c);
	}

	/*
	* 00 - sizeof(T)
	* 01 - 255 if big, 0 if little
	* 02 - 255 if floating, 0 if integral
	* 03 - 255 if signed integral 0 if unsigned integral
	*/
	template<class T> requires std::integral<T> or std::floating_point<T>
	static constexpr std::array<std::byte, 4> platform_agnostic_type_info() {
		std::array<std::byte, 4> result{};
		static_assert(sizeof(T) <= 255);

		result[0] = std::byte(sizeof(T));
		result[1] = std::endian::native == std::endian::big ? std::byte(255) : std::byte(0);
		
		if constexpr (std::floating_point<T>) {
			result[2] = std::byte(255);
		}

		if constexpr (std::integral<T>) {
			if constexpr (std::signed_integral<T>) {
				result[3] = std::byte(255);
			}
			if constexpr (std::unsigned_integral<T>) {

			}
		}
		return result;
	}
	
	
	static constexpr void to_native(std::vector<std::byte>& bytes, std::endian endianness) {
		if (endianness != std::endian::native) {
			std::reverse(bytes.begin(), bytes.end());
		}
	}

	template<class T, size_t R, size_t C>
	class MatrixView
	{
		public:
			MatrixView(std::contiguous_iterator auto it)
				: data_begin{ std::to_address(it) }
			{}
			static constexpr size_t rows() { return R; }
			static constexpr size_t cols() { return C; }

			auto operator()(std::ptrdiff_t r, std::ptrdiff_t c) {
				const size_t rv = (rows() + r) % rows();
				const size_t cv = (cols() + c) % cols();
				return matrix_at(data_begin, rv, cv, R, C);
			}

		private:
			T* data_begin;
	};


	template<typename T>
	struct FillTag
	{
		explicit FillTag(const T&v)
			: val(v)
		{}

		const T& val;
	};

	template<class T, size_t R, size_t C>
	class Matrix
	{
		public:
			explicit Matrix(const T& fill_v)
				: data(R * C, fill_v)
			{}

			explicit Matrix(std::span<const T> span, const T& fill_v = T(0))
				: data(span.begin(), span.size() >= (R * C) ? (std::next(span.begin(), R * C)) : (span.end()))
			{
				data.resize(R * C, fill_v);
			}

			explicit Matrix(std::initializer_list<T> list)
				: data(list.begin(), list.size() >= (R * C) ? (std::next(list.begin(), R * C)) : (list.end()))
			{
				data.resize(R * C, T(0));
			}

			Matrix(FillTag<T> fill_v)
				: Matrix(fill_v.val)
			{}


			explicit Matrix(const concepts::matrix_indices_function<T> auto& f)
				: data([](const auto& f) {
						std::vector<T> res;
						res.reserve(R* C);
						for (size_t i = 0u; i < R; ++i)
							for (size_t j = 0u; j < C; ++j)
								res.push_back(static_cast<T>(f(i, j)));
						return res;
					}(f))
			{}

			static constexpr size_t rows() { return R; }
			static constexpr size_t cols() { return C; }

			
			const auto& operator()(std::ptrdiff_t r, std::ptrdiff_t c) const {
				const size_t rv = (rows() + r) % rows();
				const size_t cv = (cols() + c) % cols();
				return matrix_at(data.data(), rv, cv, R, C);
			}

			auto& operator()(std::ptrdiff_t r, std::ptrdiff_t c) {
				const size_t rv = (rows() + r) % rows();
				const size_t cv = (cols() + c) % cols();
				return matrix_at(data.data(), rv, cv, R, C);
			}


			static auto random_std(const T& mean = T(0), const T& stddev = T(1)) {
				static std::random_device rd;
				static std::mt19937 gen(rd());
				std::normal_distribution<T> d(mean, stddev);
				return Matrix<T, R, C>([&](size_t i, size_t j) {
					return d(gen);
					});
			}
			
			void randomize_std(const T& mean = T(0), const T& stddev = T(1)) {
				static std::random_device rd;
				static std::mt19937 gen(rd());
				std::normal_distribution<T> d(mean, stddev);
				for (size_t i = 0; i < R; ++i)
					for (size_t j = 0; j < C; ++j)
						(*this)(i, j) = d(gen);

			}
			
			void fill(const T& val) {
				for (size_t i = 0u; i < R; ++i)
					for (size_t j = 0u; j < C; ++j)
						(*this)(i, j) = val;
			}


			auto apply(const concepts::function<T> auto& fn) const {
				return Matrix<T, R, C>(
					[this, &fn](size_t i, size_t j) {
						return fn((*this)(i, j));
					});
			}

			template<size_t nR, size_t nC>
			auto reshape() const {
				static_assert(nR * nC == R * C);
				return mv::Matrix<T, nR, nC>(std::span(this->data.begin(), this->data.end()), T(0));
			}

			
			auto transpose() const {
				return Matrix<T, C, R>(
					[this](size_t i, size_t j) {
						return (*this)(j, i);
					});
			}

			auto get_col(std::ptrdiff_t i) const {
				return Matrix<T, R, 1>(
					[this, i](size_t r, size_t c) {
						return (*this)(r, i);
					});
			}

			auto get_row(std::ptrdiff_t i) const {
				return Matrix<T, 1, C>(
					[this, i](size_t r, size_t c) {
						return (*this)(i, c);
					});
			}

			void operator+=(const Matrix<T, R, C>& other) {
				for (size_t i = 0; i < R; ++i)
					for (size_t j = 0; j < C; ++j)
						(*this)(i, j) += other(i, j);
			}

			void operator-=(const Matrix<T, R, C>& other) {
				for (size_t i = 0; i < R; ++i)
					for (size_t j = 0; j < C; ++j)
						(*this)(i, j) -= other(i, j);
			}

			void operator*=(const T& s) {
				for (size_t i = 0; i < R; ++i)
					for (size_t j = 0; j < C; ++j)
						(*this)(i, j) *= s;
			}

			void operator/=(const T& s) {
				for (size_t i = 0; i < R; ++i)
					for (size_t j = 0; j < C; ++j)
						(*this)(i, j) /= s;
			}

			std::vector<T>data;
		private:
			

			Matrix()
				: data()
			{}
	};

	template<typename T, size_t R1, size_t C1, size_t R2, size_t C2>
	auto mat_mul(const Matrix<T, R1, C1>& p, const Matrix<T, R2, C2>& q) requires (C1 == R2) {
		return Matrix<T, R1, C2>(
			[&](size_t i, size_t j) {
				auto res = T{};
				for (size_t k = 0u; k < C1; ++k) {
					res += p(i, k) * q(k, j);
				}
				return res;
			});
	}

	template<typename T, size_t R, size_t C>
	auto hadamard_product(const Matrix<T, R, C>& p, const Matrix<T, R, C>& q) {
		auto res = p;
		for (size_t i = 0; i < R; ++i)
			for (size_t j = 0; j < C; ++j)
				res(i, j) *= q(i, j);
		return res;
	}

	template<typename T, size_t R, size_t C>
	auto sum(const Matrix<T, R, C>& m) {
		auto res = T(0);
		for (size_t i = 0; i < R; ++i)
			for (size_t j = 0; j < C; ++j)
				res += m(i, j);
		return res;
	}


	template<typename T, size_t R, size_t C>
	auto operator+(const Matrix<T, R, C>& p, const Matrix<T, R, C>& q) {
		return Matrix<T, R, C>(
			[&](auto i, auto j) {
				return p(i, j) + q(i, j);
			});
	}

	template<typename T, size_t R, size_t C>
	auto operator-(const Matrix<T, R, C>& p, const Matrix<T, R, C>& q) {
		return Matrix<T, R, C>(
			[&](auto i, auto j) {
				return p(i, j) - q(i, j);
			});
	}

	template<typename T, size_t R, size_t C>
	auto operator*(const Matrix<T, R, C>& m, const T& s) {
		return Matrix<T, R, C>(
			[&](auto i, auto j) {
				return m(i, j) * s;
			});
	}

	template<typename T, size_t R, size_t C>
	auto operator*(const T& s, const Matrix<T, R, C>& m) {
		return Matrix<T, R, C>(
			[&](auto i, auto j) {
				return s * m(i, j);
			});
	}


	template<class T, size_t R, size_t C>
	std::vector<std::byte> serialize(const Matrix<T, R, C>& m) {
		static constexpr auto pa_ti_size_t = platform_agnostic_type_info<size_t>();
		static constexpr auto pa_ti_T = platform_agnostic_type_info<T>();

		std::vector<std::byte> result(m.rows() * m.cols() * sizeof(T) + 2 * sizeof(size_t) + sizeof(pa_ti_size_t) + sizeof(pa_ti_T));
		size_t offset = 0;

		std::memcpy(result.data() + offset, &pa_ti_size_t, sizeof pa_ti_size_t);
		offset += sizeof pa_ti_size_t;

		static constexpr size_t rows = m.rows();
		static constexpr size_t cols = m.cols();
		std::memcpy(result.data() + offset, &rows, sizeof(size_t));
		offset += sizeof(size_t);

		std::memcpy(result.data() + offset, &cols, sizeof(size_t));
		offset += sizeof(size_t);

		std::memcpy(result.data() + offset, &pa_ti_T, sizeof pa_ti_T);
		offset += sizeof pa_ti_T;

		for (const auto& v : m.data) {
			std::memcpy(result.data() + offset, &v, sizeof(T));
			offset += sizeof(T);
		}

		return result;
	}

	template<class T, size_t R, size_t C>
	auto deserialize(const std::vector<std::byte>& bytes) {

		size_t offset = 0;

		std::array<std::byte, 4>pa_ti_size_t{};
		std::memcpy(&pa_ti_size_t, bytes.data() + offset, sizeof pa_ti_size_t);
		offset += sizeof pa_ti_size_t;

		const auto sizeof_size_t = std::to_integer<size_t>(pa_ti_size_t[0]);
		const auto size_t_endianness = std::to_integer<bool>(pa_ti_size_t[1]) ? std::endian::big : std::endian::little;
		const bool is_integral = not std::to_integer<bool>(pa_ti_size_t[2]);
		const bool is_unsigned = not std::to_integer<bool>(pa_ti_size_t[3]);


		if (sizeof_size_t > sizeof(size_t)) {
			throw std::logic_error("data type inside file is too wide");
		}
		if (not is_unsigned) {
			throw std::logic_error("not implemented");
		}
		if (not is_integral) {
			throw std::logic_error("data type inside file was not an integer");
		}

		std::vector<std::byte>size_t_bytes(sizeof_size_t);

		std::memcpy(size_t_bytes.data(), bytes.data() + offset, sizeof_size_t);
		offset += sizeof_size_t;
		to_native(size_t_bytes, size_t_endianness);
		size_t rows{};
		std::memcpy(&rows, size_t_bytes.data(), sizeof_size_t);

		std::memcpy(size_t_bytes.data(), bytes.data() + offset, sizeof_size_t);
		offset += sizeof_size_t;
		to_native(size_t_bytes, size_t_endianness);
		size_t cols{};
		std::memcpy(&cols, size_t_bytes.data(), sizeof_size_t);


		if (rows != R or cols != C) {
			throw std::logic_error("");
		}

		std::array<std::byte, 4>pa_ti_T{};
		std::memcpy(&pa_ti_T, bytes.data() + offset, sizeof pa_ti_T);
		offset += sizeof pa_ti_T;

		const auto sizeof_T = std::to_integer<size_t>(pa_ti_T[0]);
		const auto T_endianness = std::to_integer<bool>(pa_ti_T[1]) ? std::endian::big : std::endian::little;
		const bool is_float = std::to_integer<bool>(pa_ti_T[2]);

		if (sizeof(T) != sizeof_T) {
			throw std::logic_error("data type in file has different width than data type in Matrix");
		}

		if (std::floating_point<T> != is_float) {
			throw std::logic_error("data types are different, one if float and one is not");
		}

		std::vector<std::byte>T_bytes(sizeof_T);

		auto res = mv::Matrix<T, R, C>(T(0));
				
		for (size_t i = 0; i < res.data.size(); ++i, offset += sizeof_T) {
			std::memcpy(T_bytes.data(), bytes.data() + offset, sizeof_T);
			to_native(T_bytes, T_endianness);
			std::memcpy(res.data.data() + i, T_bytes.data(), sizeof_T);
		}
		return res;
	}

	
	template<typename T, size_t R>
	using ColVector = Matrix<T, R, 1>;

	template<typename T, size_t C>
	using RowVector = Matrix<T, 1, C>;


	template<typename T, size_t R, size_t C>
	std::pair<size_t, size_t> argmax(const Matrix<T, R, C>& m) {
		size_t r = 0;
		size_t c = 0;
		for(size_t i = 0; i < R; ++i)
			for (size_t j = 0; j < C; ++j) {
				if (m(i, j) > m(r, c)) {
					r = i;
					c = j;
				}
			}

		return std::make_pair(r, c);
	}

	template<typename T, size_t R>
	size_t argmax(const ColVector<T, R>& v) {
		return argmax<T, R, 1>(v).first;
	}

	template<typename T, size_t C>
	size_t argmax(const RowVector<T, C>& v) {
		return argmax<T, 1, C>(v).second;
	}

	class MatrixFunction
	{
	public:
		MatrixFunction() = delete;
		MatrixFunction(const MatrixFunction& m) = default;
		MatrixFunction& operator=(const MatrixFunction& m) = default;
			

		template<class X, size_t R, size_t C>
		static auto get_callable(std::uint8_t fid) -> std::function<Matrix<X, R, C>(const Matrix<X, R, C>&)> {
			switch (fid) {
			case ID::identity:
				return [](const Matrix<X, R, C>& x) {
					return x;
					};
			case ID::derivative(ID::identity):
				return [](const Matrix<X, R, C>& x) {
					return Matrix<X, R, C>(FillTag(X(1)));
					};
			case ID::relu:
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							y(i, j) = std::max(X(0), y(i, j));
					return y;
					};
			case ID::derivative(ID::relu):
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							y(i, j) = y(i, j) > X(0) ? X(1) : X(0);
					return y;
					};
				break;
			case ID::logistic:
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							y(i, j) = X(1) / (X(1) + std::exp(-y(i, j)));
					return y;
					};
			case ID::derivative(ID::logistic):
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j) {
							const auto fx = X(1) / (X(1) + std::exp(-y(i, j)));
							y(i, j) = fx * (X(1) - fx);
						}
					return y;
					};
			case ID::softmax:
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					auto maxy = y(0, 0);
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							maxy = std::max(y(i, j), maxy);

					auto sumy = X(0);
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j) {
							y(i, j) = std::exp(y(i, j) - maxy);
							sumy += y(i, j);
						}
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							y(i, j) /= sumy;

					return y;
					};

			case ID::derivative(ID::softmax):
				return [](const Matrix<X, R, C>& x) {
					auto y = x;
					auto maxy = y(0, 0);
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j)
							maxy = std::max(y(i, j), maxy);

					auto sumy = X(0);
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j) {
							y(i, j) = std::exp(y(i, j) - maxy);
							sumy += y(i, j);
						}
					for (size_t i = 0; i < y.rows(); ++i)
						for (size_t j = 0; j < y.cols(); ++j) {
							const auto softmaxy = y(i, j) / sumy;
							y(i, j) = softmaxy * (X(1) - softmaxy);
						}
					return y;
					};
			
			case ID::qudratic:
				return [](const Matrix<X, R, C>& x) {
					return x.apply([](X x) {return x * x; });
					};

			case ID::derivative(ID::qudratic):
				return [](const Matrix<X, R, C>& x) {
					return x.apply([](X x) {return X(2) * x; });
					};

			case ID::cubic:
				return [](const Matrix<X, R, C>& x) {
					return x.apply([](X x) {return x * x * x; });
					};

			case ID::derivative(ID::cubic):
				return [](const Matrix<X, R, C>& x) {
					return x.apply([](X x) {return X(3) * x * x; });
					};
			default:
				throw std::logic_error("Provided nonexisting function id");
			}
			
		}

		template<class X, size_t R, size_t C>
		Matrix<X, R, C> operator()(const Matrix<X, R, C>& x) const {
			return get_callable<X, R, C>(this->id)(x);
		}

		static auto identity() {
			return MatrixFunction(ID::identity);
		}

		static auto relu() {
			return MatrixFunction(ID::relu);
		}

		static auto logistic() {
			return MatrixFunction(ID::logistic);
		}

		static auto softmax() {
			return MatrixFunction(ID::softmax);
		}

		static auto quadratic() {
			return MatrixFunction(ID::qudratic);
		}

		static auto cubic() {
			return MatrixFunction(ID::cubic);
		}

		auto derivative() const {
			return MatrixFunction(ID::derivative(this->id));
		}

	private:
		struct ID
		{
			static constexpr std::uint8_t derivative_mask = 1;

			static constexpr std::uint8_t identity = 0;
			static constexpr std::uint8_t relu = 2;
			static constexpr std::uint8_t logistic = 4;
			static constexpr std::uint8_t softmax = 6;
			static constexpr std::uint8_t qudratic = 8;
			static constexpr std::uint8_t cubic = 10;

			static constexpr std::uint8_t derivative(std::uint8_t id) { return derivative_mask | id; }
		};

		MatrixFunction(std::uint8_t id)
			: id{ id }
		{}

		std::uint8_t id;
	};

	
	
	class Random {
		
	public:
		static auto& engine() {
			static std::mt19937 gen{ std::random_device()() };
			return gen;
		}

		static void set_seed(std::uint_least32_t seed) {
			Random::engine().seed(seed);
		}

		template<std::integral Int>
		static Int uniform_distribution(Int a, Int b) {
			return std::uniform_int_distribution<Int>(a, b)(Random::engine());
		}

		template<std::floating_point Float>
		static Float uniform_distribution(Float a, Float b) {
			return std::uniform_real_distribution<Float>(a, b)(Random::engine());
		}

		template<std::floating_point Float>
		static Float normal_distribution(Float mean = Float(0), Float stddev = Float(1)) {
			return std::normal_distribution<Float>(mean, stddev)(Random::engine());
		}
	};
}