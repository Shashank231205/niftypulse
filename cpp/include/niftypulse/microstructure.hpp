// Per-instrument microstructure features: the C++ twin of
// src/niftypulse/features/microstructure.py. Same definitions, same order, checked for
// parity by tests/test_native_parity.py. No allocation on the per-event path except the
// bounded mid-price history.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace niftypulse {

constexpr double kNs = 1e9;
constexpr double kBps = 1e4;

struct Level {
    double price;
    double qty;
};

struct Book {
    int64_t ts;
    std::vector<Level> bids;  // best first
    std::vector<Level> asks;  // best first
};

struct Trade {
    int64_t ts;
    double price;
    double qty;
    int aggressor;  // +1 buy, -1 sell
};

// Sum of x_i * exp(-(t - t_i) / tau): O(1) per update, no buffers.
class DecayedSum {
public:
    explicit DecayedSum(double tau_s) : tau_ns_(tau_s * kNs) {}

    double decay_to(int64_t ts) {
        if (last_ && ts > last_) value_ *= std::exp(-static_cast<double>(ts - last_) / tau_ns_);
        last_ = std::max(last_, ts);
        return value_;
    }

    void add(double x, int64_t ts) {
        decay_to(ts);
        value_ += x;
    }

    double tau_s() const { return tau_ns_ / kNs; }

private:
    double tau_ns_;
    double value_ = 0.0;
    int64_t last_ = 0;
};

inline double ofi_level(const Level* prev, const Level* cur, bool is_bid) {
    if (!prev || !cur) return 0.0;
    if (is_bid) {
        if (cur->price > prev->price) return cur->qty;
        if (cur->price == prev->price) return cur->qty - prev->qty;
        return -prev->qty;
    }
    if (cur->price < prev->price) return -cur->qty;
    if (cur->price == prev->price) return -(cur->qty - prev->qty);
    return prev->qty;
}

inline double imbalance(const Book& b, size_t depth) {
    double bid = 0, ask = 0;
    for (size_t i = 0; i < std::min(depth, b.bids.size()); ++i) bid += b.bids[i].qty;
    for (size_t i = 0; i < std::min(depth, b.asks.size()); ++i) ask += b.asks[i].qty;
    return bid + ask > 0 ? (bid - ask) / (bid + ask) : 0.0;
}

inline double book_slope(const std::vector<Level>& levels, double mid) {
    double cum = 0, sx = 0, sy = 0;
    std::vector<std::pair<double, double>> pts;
    pts.reserve(levels.size());
    for (const auto& lv : levels) {
        const double dist = std::abs(lv.price - mid) / mid;
        cum += lv.qty;
        if (dist > 0 && cum > 0) pts.emplace_back(std::log(dist), std::log(cum));
    }
    const size_t n = pts.size();
    if (n < 3) return 0.0;
    for (const auto& [x, y] : pts) {
        sx += x;
        sy += y;
    }
    const double mx = sx / n, my = sy / n;
    double var = 0, cov = 0;
    for (const auto& [x, y] : pts) {
        var += (x - mx) * (x - mx);
        cov += (x - mx) * (y - my);
    }
    return var ? cov / var : 0.0;
}

class MicrostructureState {
public:
    MicrostructureState(std::vector<double> windows, std::vector<double> rv_windows, int ofi_levels,
                        double history_s)
        : windows_(std::move(windows)),
          rv_windows_(std::move(rv_windows)),
          ofi_levels_(ofi_levels),
          history_s_(history_s),
          depth_norm_(30.0),
          depth_weight_(30.0) {
        for (double w : windows_) {
            ofi_.emplace_back(w);
            mlofi_.emplace_back(w);
            signed_vol_.emplace_back(w);
            total_vol_.emplace_back(w);
            trade_count_.emplace_back(w);
        }
        for (double w : rv_windows_) rv_.emplace_back(w);
    }

    bool ready() const { return has_book_ && !std::isnan(mid_); }

    void on_trade(const Trade& t) {
        for (size_t i = 0; i < windows_.size(); ++i) {
            signed_vol_[i].add(t.aggressor * t.qty, t.ts);
            total_vol_[i].add(t.qty, t.ts);
            trade_count_[i].add(1.0, t.ts);
        }
        last_trade_ = t.ts;
    }

    void on_book(const Book& b) {
        if (b.bids.empty() || b.asks.empty()) return;
        const int64_t ts = b.ts;
        const Level& bid = b.bids[0];
        const Level& ask = b.asks[0];
        const double mid = 0.5 * (bid.price + ask.price);
        if (has_book_ && !last_.bids.empty() && !last_.asks.empty()) {
            const double l1 = ofi_level(&last_.bids[0], &bid, true) + ofi_level(&last_.asks[0], &ask, false);
            double multi = 0;
            for (int i = 0; i < ofi_levels_; ++i) {
                multi += ofi_level(at(last_.bids, i), at(b.bids, i), true) +
                         ofi_level(at(last_.asks, i), at(b.asks, i), false);
            }
            for (size_t i = 0; i < windows_.size(); ++i) {
                ofi_[i].add(l1, ts);
                mlofi_[i].add(multi, ts);
            }
        }
        if (!std::isnan(mid_) && mid_ > 0) {
            const double r = std::log(mid / mid_);
            for (auto& acc : rv_) acc.add(r * r, ts);
        }
        depth_norm_.add(0.5 * (bid.qty + ask.qty), ts);
        depth_weight_.add(1.0, ts);
        mid_ = mid;
        spread_ = ask.price - bid.price;
        last_ = b;
        has_book_ = true;
        mids_.emplace_back(ts, mid);
        const int64_t horizon = ts - static_cast<int64_t>(history_s_ * kNs);
        while (mids_.size() > 1 && mids_[1].first <= horizon) mids_.pop_front();
    }

    double mid_return_bps(double lookback_s, int64_t now) const {
        const int64_t target = now - static_cast<int64_t>(lookback_s * kNs);
        double past = std::numeric_limits<double>::quiet_NaN();
        for (const auto& [ts, m] : mids_) {
            if (ts > target) break;
            past = m;
        }
        if (std::isnan(past) || past <= 0 || std::isnan(mid_)) return 0.0;
        return kBps * std::log(mid_ / past);
    }

    // Appends features in the exact order of MicrostructureState.feature_names() in Python.
    void features(int64_t now, std::vector<double>& out) {
        const Level& bid = last_.bids[0];
        const Level& ask = last_.asks[0];
        const double weight = depth_weight_.decay_to(now);
        double avg_depth = weight > 0 ? depth_norm_.decay_to(now) / weight : 1.0;
        if (avg_depth == 0) avg_depth = 1.0;
        const double micro = (bid.price * ask.qty + ask.price * bid.qty) / (bid.qty + ask.qty);
        out.push_back(kBps * spread_ / mid_);
        out.push_back(kBps * (micro - mid_) / mid_);
        out.push_back(imbalance(last_, 1));
        out.push_back(imbalance(last_, 5));
        out.push_back(imbalance(last_, 10));
        out.push_back(book_slope(last_.bids, mid_));
        out.push_back(book_slope(last_.asks, mid_));
        for (auto& o : ofi_) out.push_back(o.decay_to(now) / avg_depth);
        for (auto& m : mlofi_) out.push_back(m.decay_to(now) / (avg_depth * ofi_levels_));
        for (size_t i = 0; i < windows_.size(); ++i) {
            const double total = total_vol_[i].decay_to(now);
            out.push_back(total > 0 ? signed_vol_[i].decay_to(now) / total : 0.0);
        }
        for (auto& tc : trade_count_) out.push_back(tc.decay_to(now) / tc.tau_s());
        std::vector<double> rvs;
        for (auto& acc : rv_) rvs.push_back(kBps * std::sqrt(std::max(acc.decay_to(now), 0.0)));
        out.insert(out.end(), rvs.begin(), rvs.end());
        out.push_back(rvs.back() > 0 ? rvs.front() / rvs.back() : 1.0);
        for (double w : windows_) out.push_back(mid_return_bps(w, now));
        const double since = last_trade_ ? (now - last_trade_) / kNs : history_s_;
        out.push_back(std::min(since, history_s_));
    }

    double mid() const { return mid_; }

private:
    static const Level* at(const std::vector<Level>& v, int i) {
        return i < static_cast<int>(v.size()) ? &v[i] : nullptr;
    }

    std::vector<double> windows_, rv_windows_;
    int ofi_levels_;
    double history_s_;
    double mid_ = std::numeric_limits<double>::quiet_NaN();
    double spread_ = std::numeric_limits<double>::quiet_NaN();
    Book last_{};
    bool has_book_ = false;
    int64_t last_trade_ = 0;
    std::deque<std::pair<int64_t, double>> mids_;
    std::vector<DecayedSum> ofi_, mlofi_, signed_vol_, total_vol_, trade_count_, rv_;
    DecayedSum depth_norm_, depth_weight_;
};

}  // namespace niftypulse
