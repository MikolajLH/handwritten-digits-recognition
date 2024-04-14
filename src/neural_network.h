#include "matvec.h"
#include <tuple>
#include <utility>
#include <type_traits>
#include <array>
#include <iostream>
#include <functional>

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
            auto& get_activations() { return std::get<LayerIndex>(activations); }

            auto& get_input() { return get_activations<0>(); }

            auto& get_output() { return get_activations<LayersNumber - 1>(); }


            template<size_t LayerIndex>
            auto& get_function() { return std::get<LayerIndex>(functions); }

            void for_each_weights(const auto& fn) {
                size_t index = 0u;
                auto closure_fn = [&](const auto& weights) {
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


            void backpropagate(const mv::Matrix<T, impl::get_last_element<Is...>, 1>& cost_derivative) {
                constexpr size_t L = LayersNumber - 1;

                this->get_layers_errors<L>() = mv::hadamard_product(cost_derivative, get_function<L>().derivative()(get_weighted_inputs<L>()));
                
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

}
