#pragma once
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/activations/activation_function.h"
#include <vector>

// implements a binarized fully-connected layer and "compacted" batch normalization
// pretrained only, i.e. does not support training in tiny-cnn
// use the set_threshold_from_batchnorm function for each neuron to absorb the
// batchnorm parameters into thresholds

namespace tiny_cnn {

template<typename Activation>
class binarynet_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    binarynet_layer(cnn_size_t in_dim, cnn_size_t out_dim, float_t pruneThreshold)
        : Base(in_dim, out_dim, size_t(in_dim) * out_dim, 0), pruneThreshold_(pruneThreshold) {
        // initialize all binarized weights, thresholds and output flips
        for(unsigned int i = 0; i < connection_size(); i++) {
            Wbin_.push_back(false);
            Wdisable_.push_back(false);
            Threshold_.push_back(0);
        }
    }

    // save/load -- TODO
    virtual void save(std::ostream& os) const {
        throw "Saving/loading BinaryNet layers directly not yet supported, need to load from npz";
    }

    virtual void load(std::istream& is) {
        throw "Saving/loading BinaryNet layers directly not yet supported, need to load from npz";
    }

    size_t connection_size() const override {
        // number of connections/parameters in this layer
        // - one for each synaptic weight
        // - one for each neuron's threshold
        // - one for each neuron's flip indicator
        return size_t(in_size_) * out_size_ + 2*out_size_;
    }

    size_t fan_in_size() const override {
        return in_size_;
    }

    size_t fan_out_size() const override {
        return out_size_;
    }

    virtual void post_update() {
        // once the weights have been updated, update the binarized versions too
        float2bipolar(W_, Wbin_);
        findSmallWeights(W_, Wdisable_, pruneThreshold_);
    }

    void set_threshold_from_batchnorm(size_t index, float_t mean, float_t gamma, float_t invstd, float_t beta) {
        // "compact" the batchnorm parameters into a single threshold
        // how does this work? we exploit the fact that the batchnorm parameters are constant during inference.
        // let fc = sum(weights*inputs) for a binarized neuron
        // let beta, gamma, mean and invstd be the learned batchnorm parameters
        // a binarized neuron with batch normalization and the sign activation function
        // computes the following:
        // sign(gamma*(fc - mean)*invstd+beta)
        // if we plot the neuron output as a function of the fc output, it looks like this:
        //        threshold
        //        |
        //        v
        //        |--------
        //        |
        // _______|
        // so for some value of fc (which we call the threshold), the neuron's output will change sign.
        // since the batchnorm output itself is linear, we can compute where the sign changes
        // by solving for the fc value that sets the batch-normalized output value to zero:
        // gamma*(fc - mean)*invstd+beta = 0
        // fc = mean-beta/(gamma*invstd)
        int thres = mean - (beta / (gamma*invstd));

        // depending on the sign of the multiplicative factor (gamma*invstd) the neuron output plot
        // may reverse direction, e.g:
        //        |--------           --------|
        //        |           or              |
        // _______|                           |_________
        // this could be handled by keeping an extra bit per neuron and flipping the output sign,
        // but we flip the signs of all weights and the threshold instead.
        if((gamma*invstd) < 0) {
            thres = -thres;
            for (cnn_size_t c = 0; c < in_size_; c++) {
                Wbin_[c*out_size_ + index] = !Wbin_[c*out_size_ + index];
            }
        }
        // ensure a positive threshold by averaging with the neuron fan-in
        // TODO fan-in will change per-neuron for pruned networks!
        // by ensuring a positive threshold, it becomes possible to use popcount (instead of signed add)
        // for the addition, followed by a greater than comparison for the threshold
        Threshold_[index] = (thres + fan_in_size()) / 2;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        std::vector<bool> in_bin(in_size_, false);
        // explicitly binarize the input
        float2bipolar(in, in_bin);
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            a[i] = 0;
            for (cnn_size_t c = 0; c < in_size_; c++) {
                // multiplication for binarized values is basically XNOR (equals)
                // i.e. if two values have the same sign (pos-pos or neg-neg)
                // we increment the popcount for this row
                if(!Wdisable_[c*out_size_ + i]) // if weight is pruned, don't compute
                    a[i]  += (Wbin_[c*out_size_ + i] == in_bin[c]) ? +1 : 0;
            }
            // compute the activation by comparing against the threshold
            // (the tiny-cnn specified act.fn. becomes unnecessary)
            out[i] = a[i] >= Threshold_[i] ? +1 : -1;
        });

        CNN_LOG_VECTOR(out, "[binarynet]forward");

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

    std::string layer_type() const override { return "binarynet-fully-connected"; }

protected:
    std::vector<bool> Wbin_;
    std::vector<bool> Wdisable_;
    std::vector<unsigned int> Threshold_;
    float_t pruneThreshold_;

    // utility function to convert a vector of floats into a vector of bools, where the
    // output boolean represents the sign of the input value (false: negative,
    // true: positive)
    void float2bipolar(const vec_t & in, std::vector<bool> & out) {
        for(unsigned int i = 0; i < in.size(); i++) {
            out[i] = in[i] >= 0 ? true : false;
        }
    }

    // compare a set of weights against a threshold and create a vector of bits that indicates
    // whether the weight is under the threshold (i.e. can probably by excluded from the computation)
    void findSmallWeights(const vec_t & in, std::vector<bool> & under_threshold, float_t WThres) {
        unsigned int numSmallWeights = 0;
        for(unsigned int i = 0; i < in.size(); i++) {
            float_t wabs = in[i] < 0 ? -in[i] : in[i];
            under_threshold[i] = wabs < WThres;
            numSmallWeights += wabs < WThres ? 1 : 0;
        }
        std::cout << "pruned weights " << numSmallWeights << " total weights: " << in.size() << std::endl;
        // TODO maybe compute a neuron output threshold update during this?
    }

};

} // namespace tiny_cnn
