#pragma once
#include <vector>
#include <iterator>
#include <cassert>
#include <type_traits>
#include <cmath>
#include <functional>
#include <span>
#include <random>
#include <initializer_list>

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

	

	template<typename T, size_t R, size_t K, size_t C>
	auto mat_mul(const Matrix<T, R, K>& p, const Matrix<T, K, C>& q) {
		return Matrix<T, R, C>(
			[&](auto i, auto j) {
				auto res = T{};
				for (size_t k = 0u; k < K; ++k) {
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
		std::vector<std::byte> result(m.rows() * m.cols() * sizeof(T) + 2 * sizeof(size_t));
		size_t offset = 0;
		const size_t rows = m.rows();
		const size_t cols = m.cols();
		std::memcpy(result.data() + offset, &rows, sizeof(size_t));
		offset += sizeof(size_t);

		std::memcpy(result.data() + offset, &cols, sizeof(size_t));
		offset += sizeof(size_t);

		for (const auto& v : m.data) {
			std::memcpy(result.data() + offset, &v, sizeof(T));
			offset += sizeof(T);
		}

		return result;
	}

	template<class T, size_t R, size_t C>
	auto deserialize(const std::vector<std::byte>& bytes) {
		auto res = mv::Matrix<T, R, C>(T(0));
		size_t rows;
		size_t cols;
		size_t offset = 0;
		std::memcpy(&rows, bytes.data() + offset, sizeof(size_t));
		offset += sizeof(size_t);

		std::memcpy(&cols, bytes.data() + offset, sizeof(size_t));
		offset += sizeof(size_t);

		for (size_t i = 0; i < res.data.size(); ++i, offset += sizeof(T)) {
			std::memcpy(res.data.data() + i, bytes.data() + offset, sizeof(T));
		}
		return res;
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
}