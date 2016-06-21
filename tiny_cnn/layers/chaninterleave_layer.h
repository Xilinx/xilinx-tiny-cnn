#pragma once
#include "tiny_cnn/util/util.h"

// each channel's feature maps are either contiguous or interleaved,
// and this layer switches between those two modes. does not touch
// row major / column major order of the pixels.

namespace tiny_cnn {


template<typename Activation>
class chaninterleave_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef layer<Activation> Base;

    explicit chaninterleave_layer(cnn_size_t channels, cnn_size_t pixelsPerChan, bool deinterleave)
        : Base(channels*pixelsPerChan, channels*pixelsPerChan, 0, 0), channels_(channels),
          pixelsPerChan_(pixelsPerChan), deinterleave_(deinterleave)
    {

    }

    size_t param_size() const override {
        return 0;
    }

    size_t connection_size() const override {
        return this->in_size();
    }

    size_t fan_in_size() const override {
        return 1;
    }

    size_t fan_out_size() const override {
        return 1;
    }

    std::string layer_type() const override { return "chaninterleave_layer"; }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& a = a_[index];
        vec_t& out = output_[index];

        for(unsigned int c = 0; c < channels_; c++) {
            for(unsigned int pix = 0; pix < pixelsPerChan_; pix++) {
                if(deinterleave_) {
                    out[c * pixelsPerChan_ + pix] = in[pix*channels_ + c];
                } else {
                    out[pix*channels_ + c] = in[c * pixelsPerChan_ + pix];
                }
            }
        }

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        throw "Not implemented";

        return current_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not implemented";

        return current_delta2;
    }

protected:
    cnn_size_t channels_, pixelsPerChan_;
    bool deinterleave_;
};

} // namespace tiny_cnn
