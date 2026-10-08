// Python bindings (pybind11) for the hot path: niftypulse_native.MicrostructureState and
// niftypulse_native.Mlp. Built when pybind11 and a CPython-compatible compiler are
// available (CI builds it with MSVC and GCC); the pure-Python engine remains the
// reference implementation.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "niftypulse/microstructure.hpp"
#include "niftypulse/mlp.hpp"

namespace py = pybind11;
using namespace niftypulse;

namespace {

Book to_book(int64_t ts, const std::vector<std::pair<double, double>>& bids,
             const std::vector<std::pair<double, double>>& asks) {
    Book b;
    b.ts = ts;
    for (const auto& [p, q] : bids) b.bids.push_back({p, q});
    for (const auto& [p, q] : asks) b.asks.push_back({p, q});
    return b;
}

}  // namespace

PYBIND11_MODULE(niftypulse_native, m) {
    m.doc() = "C++ feature engine and MLP inference for NiftyPulse";

    py::class_<MicrostructureState>(m, "MicrostructureState")
        .def(py::init<std::vector<double>, std::vector<double>, int, double>())
        .def("on_book",
             [](MicrostructureState& s, int64_t ts, const std::vector<std::pair<double, double>>& bids,
                const std::vector<std::pair<double, double>>& asks) { s.on_book(to_book(ts, bids, asks)); })
        .def("on_trade",
             [](MicrostructureState& s, int64_t ts, double price, double qty, int aggressor) {
                 s.on_trade(Trade{ts, price, qty, aggressor});
             })
        .def("ready", &MicrostructureState::ready)
        .def("features", [](MicrostructureState& s, int64_t now) {
            std::vector<double> out;
            s.features(now, out);
            return out;
        });

    py::class_<Mlp>(m, "Mlp")
        .def(py::init<const std::string&>())
        .def("inputs", &Mlp::inputs)
        .def("predict", [](Mlp& mlp, const std::vector<double>& x) {
            if (x.size() != mlp.inputs()) throw std::invalid_argument("wrong feature count");
            std::vector<double> out(3);
            mlp.predict(x.data(), out.data());
            return out;
        });
}
