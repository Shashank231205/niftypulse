// Replays an event file through the C++ feature engine, writes the features for parity
// checks, and reports latency (JSON on stdout):
//   - feature update + extraction per book event (p50/p99, ns)
//   - order-book level updates (synthetic, ns/update)
//   - student MLP inference (ns/prediction), if a weight file is given
//
// usage: replay_bench <events.npev> <features.out> <windows> <rv_windows> <ofi_levels>
//                     <history_s> [mlp.npmlp]
// Event file, little-endian: "NPEV1", u32 depth, then records:
//   u8 kind (0 book, 1 trade), i64 ts,
//   book: u16 n_bids, u16 n_asks, (f64 price, f64 qty) * (n_bids + n_asks)
//   trade: f64 price, f64 qty, i8 aggressor
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "niftypulse/microstructure.hpp"
#include "niftypulse/mlp.hpp"
#include "niftypulse/order_book.hpp"

using namespace niftypulse;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<double> parse_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
    return out;
}

template <typename T>
bool read(std::ifstream& f, T& v) {
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    const size_t k = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

double bench_order_book(size_t updates) {
    OrderBook book;
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> tick(-50, 50);
    std::uniform_real_distribution<double> qty(0.0, 5.0);
    std::bernoulli_distribution side(0.5), remove(0.2);
    Book out;
    const auto t0 = Clock::now();
    for (size_t i = 0; i < updates; ++i) {
        const bool bid = side(rng);
        const double px = 100000.0 + (bid ? -1 : 1) * (1 + std::abs(tick(rng))) * 0.1;
        book.apply(bid, px, remove(rng) ? 0.0 : qty(rng));
        if (i % 10 == 0) book.top(20, out);
    }
    const auto ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    return ns / static_cast<double>(updates);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: replay_bench <events> <features_out> <windows> <rv_windows> "
                     "<ofi_levels> <history_s> [mlp]\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    char magic[5];
    uint32_t depth = 0;
    if (!in.read(magic, 5) || std::memcmp(magic, "NPEV1", 5) != 0 || !read(in, depth)) {
        std::cerr << "bad event file\n";
        return 1;
    }
    MicrostructureState state(parse_list(argv[3]), parse_list(argv[4]), std::stoi(argv[5]),
                              std::stod(argv[6]));

    std::vector<double> row, all;
    std::vector<double> book_ns;
    uint32_t n_cols = 0, n_rows = 0;
    size_t trades = 0;
    Book book;
    uint8_t kind;
    int64_t ts;
    while (read(in, kind) && read(in, ts)) {
        if (kind == 0) {
            uint16_t nb, na;
            read(in, nb);
            read(in, na);
            book.ts = ts;
            book.bids.resize(nb);
            book.asks.resize(na);
            for (auto& lv : book.bids) { read(in, lv.price); read(in, lv.qty); }
            for (auto& lv : book.asks) { read(in, lv.price); read(in, lv.qty); }
            const auto t0 = Clock::now();
            state.on_book(book);
            row.clear();
            if (state.ready()) state.features(ts, row);
            book_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
            if (!row.empty()) {
                n_cols = static_cast<uint32_t>(row.size());
                all.insert(all.end(), row.begin(), row.end());
                ++n_rows;
            }
        } else {
            Trade t{ts, 0, 0, 0};
            int8_t agg;
            read(in, t.price);
            read(in, t.qty);
            read(in, agg);
            t.aggressor = agg;
            state.on_trade(t);
            ++trades;
        }
    }

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n_rows), 4);
    out.write(reinterpret_cast<const char*>(&n_cols), 4);
    out.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(all.size() * 8));

    double mlp_p50 = -1, mlp_p99 = -1;
    if (argc > 7) {
        Mlp mlp(argv[7]);
        std::vector<double> x(mlp.inputs(), 0.1), y(3);
        std::vector<double> times;
        for (int i = 0; i < 1000; ++i) mlp.predict(x.data(), y.data());
        for (int i = 0; i < 20000; ++i) {
            x[i % x.size()] = 0.001 * i;
            const auto t0 = Clock::now();
            mlp.predict(x.data(), y.data());
            times.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
        }
        mlp_p50 = percentile(times, 50);
        mlp_p99 = percentile(times, 99);
    }

    std::printf(
        "{\"book_events\": %zu, \"trades\": %zu, \"feature_rows\": %u, \"features\": %u, "
        "\"feature_update_ns_p50\": %.1f, \"feature_update_ns_p99\": %.1f, "
        "\"order_book_update_ns\": %.1f, \"mlp_ns_p50\": %.1f, \"mlp_ns_p99\": %.1f}\n",
        book_ns.size(), trades, n_rows, n_cols, percentile(book_ns, 50), percentile(book_ns, 99),
        bench_order_book(1'000'000), mlp_p50, mlp_p99);
    return 0;
}
