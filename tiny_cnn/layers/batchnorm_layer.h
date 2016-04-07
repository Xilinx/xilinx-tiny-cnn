#pragma ONCE
#include "tiny_cnn/layers/layer.h"

// right now functional during inference only
// pre-trained batchnorm params must be manually set
// TODO:
// - implement proper batch normalization during training

// during training, for each neuron output, batch normalization learns
// four statistics from the data that passes through it. we keep all
// four of these as part of the layer's weights, in the following order:
// * all of the shifts (beta)
// * all of the scales (gamma)
// * all of the means (mu)
// * all of the inverted stddevs (inv_std)

namespace tiny_cnn {
template<typename Activation>
class batchnorm_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    batchnorm_layer(cnn_size_t dim)
        : Base(dim, dim, 4*dim, 0) {}

    size_t connection_size() const override {
        return in_size_;
    }

    size_t fan_in_size() const override {
        return 1;
    }

    size_t fan_out_size() const override {
        return 1;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            a[i] = gamma(i) * (in[i] - mean(i)) * invstd(i) + beta(i);
        });

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });
        CNN_LOG_VECTOR(out, "[bn]forward");

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not yet implemented";
        return curr_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not yet implemented";
        return current_delta2;
    }

    std::string layer_type() const override { return "batchnorm"; }

protected:
    inline float_t beta(unsigned int ind) {
        return W_[(in_size_ * 0) + ind];
    }

    inline float_t gamma(unsigned int ind) {
        return W_[(in_size_ * 1) + ind];
    }

    inline float_t mean(unsigned int ind) {
        return W_[(in_size_ * 2) + ind];
    }

    inline float_t invstd(unsigned int ind) {
        return W_[(in_size_ * 3) + ind];
    }

};

} // namespace tiny_cnn
