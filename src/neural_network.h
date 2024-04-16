#include "matvec.h"
#include <tuple>
#include <utility>
#include <type_traits>
#include <array>
#include <iostream>
#include <functional>
#include <fstream>
#include <filesystem>
#include <ranges>
#include <format>
#include <chrono>

namespace nn
{
    namespace impl
    {
        template<size_t Index, size_t First, size_t...Rest>
        struct NthElementImpl {
            static constexpr size_t value = NthElementImpl<Index - 1, Rest...>::value;
        };

        template<size_t First, size_t... Rest>
        struct NthElementImpl<0u, First, Rest...>{
            static constexpr size_t value = First;
        };

        template<size_t Index, size_t...Is>
        static constexpr auto get_nth_element = NthElementImpl<Index, Is...>::value;

        template<size_t First, size_t... Rest>
        struct LastElementImpl {
            static constexpr size_t value = LastElementImpl<Rest...>::value;
        };

        template<size_t First, size_t Last>
        struct LastElementImpl<First, Last> {
            static constexpr size_t value = Last;
        };

        template<size_t...Is>
        static constexpr auto get_last_element = LastElementImpl<Is...>::value;

        template<size_t First, size_t... Rest>
        struct FirstElementImpl {
            static constexpr size_t value = First;
        };

        template<size_t...Is>
        static constexpr auto get_first_element = FirstElementImpl<Is...>::value;
        

        template<size_t First, size_t Second>
        struct Pair
        {
            static constexpr size_t first = First;
            static constexpr size_t second = Second;
        };

        template<size_t... Is>
        struct PairSequenceImpl {};

        template<size_t First, size_t Second, size_t... Rest>
        struct PairSequenceImpl<First, Second, Rest...>
        {
            using type = decltype(std::tuple_cat(
                std::tuple<Pair<First, Second>>{},
                typename PairSequenceImpl<Second, Rest...>::type{}
            ));
        };

        template<size_t Last>
        struct PairSequenceImpl<Last>
        {
            using type = std::tuple<>;
        };

        template<size_t... Is>
        using PairSequence = PairSequenceImpl<Is...>::type;


        template<typename T, class PS>
        struct WeightsImpl
        {
            using type = decltype(
                []<size_t...Is>
                (std::index_sequence<Is...>) -> std::tuple<mv::Matrix<T, std::tuple_element_t<Is, PS>::second, std::tuple_element_t<Is, PS>::first>...>
            {}(std::make_index_sequence<std::tuple_size_v<PS>>{}));
        };


        template<class T, size_t... Is>
        using Weights = typename WeightsImpl<T, PairSequence<Is...>>::type;

        template<class T, size_t... Is>
        using Layers = std::tuple<mv::Matrix<T, Is, 1 >...>;


        template<class T, size_t I>
        using IgnoreI = T;

        template<class T, size_t...Is>
        using Functions = std::tuple<IgnoreI<mv::MatrixFunction, Is>...>;

        
        
    }

    using impl::Weights;
    using impl::Layers;
    using impl::Functions;

    using impl::get_nth_element;


    template<typename T, size_t... Is>
    class MLP
    {
        public:
            static constexpr size_t InputSize = get_nth_element<0, Is...>;
            static constexpr size_t OutputSize = get_nth_element<sizeof...(Is) - 1, Is...>;
            static constexpr size_t LayersNumber = sizeof...(Is);
            static constexpr size_t WeightsNumber = sizeof...(Is) - 1;
            using ContainedType = T;


            template<size_t Index>
            static constexpr size_t weights_rows() { return std::tuple_element_t<Index, impl::PairSequence<Is...>>::second; }

            template<size_t Index>
            static constexpr size_t weights_cols() { return std::tuple_element_t<Index, impl::PairSequence<Is...>>::first; }

            template<size_t Index>
            static constexpr size_t layer_size() { return get_nth_element<Index, Is...>; }


            explicit MLP(mv::FillTag<T> fill_w)
                :
                weights(
                    [] <size_t...I>(mv::FillTag<T> fill_w, std::index_sequence<I...>) {
                return std::make_tuple(mv::Matrix<T, weights_rows<I>(), weights_cols<I>()>(fill_w)...);
            }(fill_w, std::make_index_sequence<WeightsNumber>{})),

                weighted_inputs(
                    []<size_t...Indices>(std::index_sequence<Indices...>) {
                return std::make_tuple(mv::Matrix<T, Indices, 1>(T(0))...);
            }(std::index_sequence<Is...>{})),

                activations(
                    []<size_t...Indices>(std::index_sequence<Indices...>) {
                return std::make_tuple(mv::Matrix<T, Indices, 1>(T(0))...);
            }(std::index_sequence<Is...>{})),

                layers_errors(
                    []<size_t...Indices>(std::index_sequence<Indices...>) {
                return std::make_tuple(mv::Matrix<T, Indices, 1>(T(0))...);
            }(std::index_sequence<Is...>{})),

                weights_deltas(
                    [] <size_t...I>(std::index_sequence<I...>) {
                return std::make_tuple(mv::Matrix<T, weights_rows<I>(), weights_cols<I>()>(mv::FillTag(T(0)))...);
            }(std::make_index_sequence<WeightsNumber>{})),

                functions(
                    []<size_t...Indices>(std::index_sequence<Indices...>) {
                return std::make_tuple((Indices, mv::MatrixFunction::identity())...);
            }(std::index_sequence<Is...>{}))

            {}

            template<size_t Index>
            auto& get_weights() { return std::get<Index>(weights); }

            template<size_t Index>
            auto& get_layers_errors() { return std::get<Index>(layers_errors); }

            template<size_t Index>
            auto& get_weights_deltas() { return std::get<Index>(weights_deltas); }

            template<size_t LayerIndex>
            auto& get_weighted_inputs() { return std::get<LayerIndex>(weighted_inputs); }

            template<size_t LayerIndex>
            auto& get_weighted_inputs() const { return std::get<LayerIndex>(weighted_inputs); }

            template<size_t LayerIndex>
            auto& get_activations() { return std::get<LayerIndex>(activations); }

            template<size_t LayerIndex>
            auto& get_activations() const { return std::get<LayerIndex>(activations); }

            auto& get_input() { return get_activations<0>(); }

            auto& get_output() { return get_activations<LayersNumber - 1>(); }


            template<size_t LayerIndex>
            auto& get_function() { return std::get<LayerIndex>(functions); }

            template<size_t LayerIndex>
            auto& get_function() const { return std::get<LayerIndex>(functions); }

            void for_each_weights(const auto& fn) {
                size_t index = 0u;
                auto closure_fn = [&](auto& weights) {
                    fn(weights, index++); };

                [&]<size_t...I>(std::index_sequence<I...>){
                    (closure_fn(get_weights<I>()), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void randomize() {
                [this] <size_t...I>(std::index_sequence<I...>) {
                    ((get_weights<I>().randomize_std(), 0), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void feed_forward() {
                [this] <size_t...I>(std::index_sequence<I...>) {
                    (this->calc_weighted_inputs_and_activations<I + 1>(), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void reset_layers_error() {
                [this] <size_t...I>(std::index_sequence<I...>) {
                    ((this->get_layers_errors<I>() = mv::FillTag(T())), ...);
                }(std::make_index_sequence<LayersNumber>{});
            }

            void reset_weights_deltas() {
                [this] <size_t...I>(std::index_sequence<I...>) {
                    ((this->get_weights_deltas<I>() = mv::FillTag(T())), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void reset_errors() {
                this->reset_layers_error();
                this->reset_weights_deltas();
            }


            void backpropagate(const mv::Matrix<T, OutputSize, 1>& last_layer_error) {
                constexpr size_t L = LayersNumber - 1;
                this->get_layers_errors<L>() = last_layer_error;
                
                [this] <size_t...I>(std::index_sequence<I...>) {
                    (this->calc_error_in_prev_layer<WeightsNumber - I>(), ...);

                    (this->calc_weights_deltas<I>(), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void apply_weights_deltas() {
                [this] <size_t...I>(std::index_sequence<I...>) {
                    ((this->get_weights<I>() -= this->get_weights_deltas<I>()), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void multiply_weights_deltas(const T& s) {
                [this, &s] <size_t...I>(std::index_sequence<I...>) {
                    ((this->get_weights_deltas<I>() *= s), ...);
                }(std::make_index_sequence<WeightsNumber>{});
            }

            void save(std::string_view path) {
                
                static constexpr auto pa_ti_size_t = mv::platform_agnostic_type_info<size_t>();
                static constexpr auto pa_ti_T = mv::platform_agnostic_type_info<T>();

                const size_t n_elements = WeightsNumber;
                std::vector<std::byte>header(sizeof(pa_ti_size_t) + sizeof(pa_ti_T) + sizeof(size_t) + n_elements * sizeof(size_t) * 2);
                size_t offset = 0u;
                
                std::memcpy(header.data() + offset, &pa_ti_size_t, sizeof pa_ti_size_t);
                offset += sizeof pa_ti_size_t;

                std::memcpy(header.data() + offset, &pa_ti_T, sizeof pa_ti_T);
                offset += sizeof pa_ti_T;


                
                std::memcpy(header.data() + offset, &n_elements, sizeof(size_t));
                offset += sizeof(size_t);
                for_each_weights([&](const auto& m, size_t) {
                    constexpr size_t rows = m.rows();
                    constexpr size_t cols = m.cols();
                    std::memcpy(header.data() + offset, &rows, sizeof(size_t));
                    offset += sizeof(size_t);
                    std::memcpy(header.data() + offset, &cols, sizeof(size_t));
                    offset += sizeof(size_t);
                    });

                std::ofstream file(std::filesystem::path(path), std::ios::binary);
                file.write((char*)header.data(), header.size());

                for_each_weights([&](const auto& m, size_t) {
                    const auto bytes = mv::serialize(m);
                    file.write((char*)bytes.data(), bytes.size());
                    });
            }

            void load(std::string_view path) {
                std::ifstream file(std::filesystem::path(path), std::ios::binary);

                std::array<std::byte, 4>pa_ti_size_t{};
                file.read((char*)&pa_ti_size_t, sizeof(pa_ti_size_t));

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

                std::array<std::byte, 4>pa_ti_T{};
                file.read((char*)&pa_ti_T, sizeof(pa_ti_T));

                const auto sizeof_T = std::to_integer<size_t>(pa_ti_T[0]);
                const auto T_endianness = std::to_integer<bool>(pa_ti_T[1]) ? std::endian::big : std::endian::little;
                const bool is_float = std::to_integer<bool>(pa_ti_T[2]);

                if (sizeof(T) != sizeof_T) {
                    throw std::logic_error("data type in file has different width than data type in Matrix");
                }

                if (std::floating_point<T> != is_float) {
                    throw std::logic_error("data types are different, one if float and one is not");
                }


                std::vector<std::byte>size_t_bytes(sizeof_size_t);
                file.read((char*)size_t_bytes.data(), sizeof_size_t);
                mv::to_native(size_t_bytes, size_t_endianness);
                size_t elements_n{};

                std::memcpy(&elements_n, size_t_bytes.data(), sizeof_size_t);

                file.seekg(elements_n * 2 * sizeof_size_t, std::ios_base::cur);
                for_each_weights([&]<typename T, size_t mR, size_t mC>(mv::Matrix<T, mR, mC>& m, size_t) {
                    std::vector<std::byte>bytes((mR * mC) * sizeof_T + 2 * sizeof_size_t + 4 + 4);
                    file.read((char*)bytes.data(), bytes.size());
                    m = mv::deserialize<T, mR, mC>(bytes);
                });
            }

        private:
            Weights<T, Is...> weights;
            Layers<T, Is...> weighted_inputs;
            Layers<T, Is...> activations;
            Functions<T, Is...> functions;

            Layers<T, Is...> layers_errors;
            Weights<T, Is...> weights_deltas;

            template<size_t I>
            bool calc_weighted_inputs_and_activations() {
                static_assert(I > 0 and I < LayersNumber);
                get_weighted_inputs<I>() = mv::mat_mul(get_weights<I - 1>(), get_activations<I - 1>());
                get_activations<I>() = get_function<I>()(get_weighted_inputs<I>());
                return true;
            }

            template<size_t I>
            bool calc_error_in_prev_layer() {
                get_layers_errors<I - 1>() += mv::hadamard_product(
                    mv::mat_mul(get_weights<I - 1>().transpose(), get_layers_errors<I>()),
                    get_function<I - 1>().derivative()(get_weighted_inputs<I - 1>()));
                return true;
            }

            template<size_t I>
            bool calc_weights_deltas() {
                get_weights_deltas<I>() += mv::mat_mul(get_layers_errors<I + 1>(), get_activations<I>().transpose());
                return true;
            }
    };

    class CostFunction
    {
    public:
        struct CrossEntropy {
            template<typename T, size_t...Is>
            static auto delta(
                const MLP<T, Is...>& mlp,
                const mv::Matrix<T, impl::get_last_element<Is...>, 1>& Y) {
                return mlp.get_activations<sizeof...(Is) - 1>() - Y;
            }
            template<typename T, size_t N>
            static auto fn(const mv::Matrix<T, N, 1>& A, const mv::Matrix<T, N, 1>& Y) {
                return mv::sum(
                    mv::Matrix<T, N, 1>(
                        [&](size_t r, size_t c) {
                            const auto ret = -Y(r, c) * std::log(A(r, c)) - (T(1) - Y(r, c)) * std::log(T(1) - A(r, c));
                            switch (std::fpclassify(ret)) {
                            case FP_INFINITE:
                                return T(10'000'000'000);
                            case FP_NAN:
                                return T(0);
                            default:
                                return ret;
                            }
                        }));
            }
        };

        struct Quadratic {
            template<typename T, size_t...Is>
            static auto delta(
                const MLP<T, Is...>& mlp,
                const mv::Matrix<T, impl::get_last_element<Is...>, 1>& Y) {
                return mv::hadamard_product(
                    mlp.get_activations<sizeof...(Is) - 1>() - Y,
                    mlp.get_function<sizeof...(Is) - 1>().derivative(mlp.get_weighted_inputs<sizeof...(Is) - 1>())
                );
            }
            template<typename T, size_t N>
            static auto fn(const mv::Matrix<T, N, 1>& A, const mv::Matrix<T, N, 1>& Y) {
                const auto W = A - Y;
                return T(0.5) * mv::mat_mul(W.transpose(), W);
            }
        };
    };


    template<class T, size_t InputSize, size_t OutputSize>
    using DataSet = std::vector<std::pair<mv::ColVector<T, InputSize>, mv::ColVector<T, OutputSize>>>;


    auto accuracy(auto& mlp, const auto& dataset) {
        using MLP_T = std::remove_cvref_t<decltype(mlp)>;
        using T = MLP_T::ContainedType;
        size_t correct_predictions = 0;
        for (const auto& [X, Y] : dataset) {
            mlp.get_input() = X;
            mlp.feed_forward();
            const auto Y_hat = mlp.get_output();
            correct_predictions += (mv::argmax(Y_hat) == mv::argmax(Y));
        }
        return T(correct_predictions) / T(dataset.size());
    }


    template<class CostFn, class T, size_t InputLayerSize, size_t OutputLayerSize, size_t... HiddenLayersSizes>
    void SGD(
        MLP<T, InputLayerSize, HiddenLayersSizes..., OutputLayerSize>&mlp,
        size_t epochs_number,
        DataSet<T, InputLayerSize, OutputLayerSize>& training_set,
        DataSet<T, InputLayerSize, OutputLayerSize>& validation_set,
        const T& eta,
        size_t batch_size,
        size_t early_stopping_condition,
        std::string_view model_name
        ) {

        size_t validation_accuracy_decreased = 0;
        auto last_validation_accuracy = T(0);
        auto best_validation_accuracy = T(0);

        for (size_t epoch = 0; epoch < epochs_number; ++epoch) {
            std::shuffle(training_set.begin(), training_set.end(), mv::Random::engine());
            auto epoch_cost_fn = T(0);
            size_t correct_predictions = 0u;
            const auto epoch_start = std::chrono::steady_clock::now();
            for (const auto& batch : training_set | std::views::chunk(batch_size)) {
                auto batch_cost_fn = T(0);
                mlp.reset_errors();

                for (const auto& [X, Y] : batch) {
                    mlp.get_input() = X;
                    mlp.feed_forward();
                    const auto Y_hat = mlp.get_output();

                    correct_predictions += (mv::argmax(Y_hat) == mv::argmax(Y));

                    const auto Cf = CostFn::fn(Y_hat, Y);
                    batch_cost_fn += Cf;

                    const auto delta = CostFn::delta(mlp, Y);
                    mlp.backpropagate(delta);
                }
                batch_cost_fn /= T(batch_size);
                epoch_cost_fn += batch_cost_fn;

                mlp.multiply_weights_deltas((eta) / (T(batch_size)));
                mlp.apply_weights_deltas();
            }
            epoch_cost_fn /= (T(training_set.size()) / T(batch_size));

            const T validation_accuracy = accuracy(mlp, validation_set);

            
            const T training_accuracy = T(correct_predictions) / T(training_set.size());
            correct_predictions = 0;

            if (validation_accuracy > best_validation_accuracy) {
                best_validation_accuracy = validation_accuracy;
                if (model_name.size())
                    mlp.save(model_name);
            }

            if (epoch != 0 and validation_accuracy < last_validation_accuracy) {
                ++validation_accuracy_decreased;
            }
            else {
                validation_accuracy_decreased = 0;
            }
            
            std::cout <<
                std::format(
                    "Epoch: {:4}.\ncost fn: {:8.4e}, accuracy: {:5.2f}%, best_accuracy: {:5.2f}%, training_accuracy: {:5.2f}%, early_stopping: {:>2}/{:<2}.",
                    epoch, epoch_cost_fn, T(100) * validation_accuracy, T(100) * best_validation_accuracy, T(100) * training_accuracy, validation_accuracy_decreased, early_stopping_condition) << "\n";

            if (validation_accuracy_decreased == early_stopping_condition) {
                std::cout << "Early stopping triggered.\n";
                return;
            }
            last_validation_accuracy = validation_accuracy;
            const auto epoch_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::ratio<60>> per_epoch = (epoch_end - epoch_start);
            //const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(per_epoch).count();

            std::cout << std::format("Epoch took {:4.2f}min.\nEstimated remaining training time: {:6.2f}min.\n\n",
                per_epoch.count(),
                (per_epoch * (epochs_number - epoch)).count());
        }
    }
}
