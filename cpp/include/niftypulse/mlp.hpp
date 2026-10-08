// Inference for the distilled student MLP exported by models/distillation.py
// (export_cpp_weights). Format, little-endian:
//   "NPMLP1", u32 n_layers, per layer {u32 rows, u32 cols, f32 W[rows*cols], f32 b[rows]},
//   f32 mu[F], f32 sd[F]
// ReLU between layers, softmax on the output. Buffers are preallocated: no allocation
// per prediction.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace niftypulse {

class Mlp {
public:
    explicit Mlp(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        char magic[6];
        f.read(magic, 6);
        if (std::memcmp(magic, "NPMLP1", 6) != 0) throw std::runtime_error("bad magic in " + path);
        uint32_t n = read<uint32_t>(f);
        for (uint32_t l = 0; l < n; ++l) {
            Layer layer;
            layer.rows = read<uint32_t>(f);
            layer.cols = read<uint32_t>(f);
            layer.w.resize(static_cast<size_t>(layer.rows) * layer.cols);
            layer.b.resize(layer.rows);
            f.read(reinterpret_cast<char*>(layer.w.data()), layer.w.size() * sizeof(float));
            f.read(reinterpret_cast<char*>(layer.b.data()), layer.b.size() * sizeof(float));
            layers_.push_back(std::move(layer));
        }
        const size_t in = layers_.front().cols;
        mu_.resize(in);
        sd_.resize(in);
        f.read(reinterpret_cast<char*>(mu_.data()), in * sizeof(float));
        f.read(reinterpret_cast<char*>(sd_.data()), in * sizeof(float));
        if (!f) throw std::runtime_error("truncated weight file " + path);
        size_t widest = in;
        for (const auto& layer : layers_) widest = std::max<size_t>(widest, layer.rows);
        a_.resize(widest);
        b_.resize(widest);
    }

    size_t inputs() const { return mu_.size(); }

    // x: raw (unscaled) features. out: 3 probabilities (down, flat, up).
    void predict(const double* x, double* out) {
        const size_t in = mu_.size();
        for (size_t i = 0; i < in; ++i)
            a_[i] = std::clamp(static_cast<float>((x[i] - mu_[i]) / sd_[i]), -10.0f, 10.0f);
        float* src = a_.data();
        float* dst = b_.data();
        for (size_t l = 0; l < layers_.size(); ++l) {
            const Layer& L = layers_[l];
            for (uint32_t r = 0; r < L.rows; ++r) {
                const float* w = &L.w[static_cast<size_t>(r) * L.cols];
                float acc = L.b[r];
                for (uint32_t c = 0; c < L.cols; ++c) acc += w[c] * src[c];
                dst[r] = (l + 1 < layers_.size()) ? std::max(acc, 0.0f) : acc;
            }
            std::swap(src, dst);
        }
        const float m = std::max({src[0], src[1], src[2]});
        double total = 0;
        for (int k = 0; k < 3; ++k) total += out[k] = std::exp(static_cast<double>(src[k] - m));
        for (int k = 0; k < 3; ++k) out[k] /= total;
    }

private:
    struct Layer {
        uint32_t rows = 0, cols = 0;
        std::vector<float> w, b;
    };

    template <typename T>
    static T read(std::ifstream& f) {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        return v;
    }

    std::vector<Layer> layers_;
    std::vector<float> mu_, sd_, a_, b_;
};

}  // namespace niftypulse
