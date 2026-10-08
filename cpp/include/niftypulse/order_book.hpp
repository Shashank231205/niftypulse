// Price-level book for the hot path. Mirrors marketdata/order_book.py: absolute level
// quantities, zero deletes, top-N extraction, crossed-book check.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "microstructure.hpp"

namespace niftypulse {

class OrderBook {
public:
    void apply(bool is_bid, double price, double qty) {
        if (is_bid) {
            if (qty > 0) bids_[price] = qty; else bids_.erase(price);
        } else {
            if (qty > 0) asks_[price] = qty; else asks_.erase(price);
        }
    }

    bool crossed() const {
        return !bids_.empty() && !asks_.empty() && bids_.begin()->first >= asks_.begin()->first;
    }

    void top(size_t depth, Book& out) const {
        out.bids.clear();
        out.asks.clear();
        for (auto it = bids_.begin(); it != bids_.end() && out.bids.size() < depth; ++it)
            out.bids.push_back({it->first, it->second});
        for (auto it = asks_.begin(); it != asks_.end() && out.asks.size() < depth; ++it)
            out.asks.push_back({it->first, it->second});
    }

    size_t size() const { return bids_.size() + asks_.size(); }

private:
    std::map<double, double, std::greater<>> bids_;
    std::map<double, double> asks_;
};

}  // namespace niftypulse
