#include "prune-policy.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

static bool exercise() {
    std::map<int, prune_stats::layer> stats;
    for (int L : {1, 2, 10}) {
        auto & layer = stats[L];
        layer.top_k = 2;
        layer.cnt.assign(64, 100);
        layer.msan.resize(64);
        for (int e = 0; e < 64; ++e) layer.msan[e] = (float) (e + 1);
    }
    auto keep = prune_policy::select(stats, {1, -1, -1.0, -1});
    if (keep.size() != stats.size()) return false;
    for (const auto & item : keep)
        if (item.second != std::vector<int>({62, 63})) return false;
    if (!prune_policy::select(stats, {64, -1, -1.0, -1}).empty()) return false;
    keep = prune_policy::select(stats, {-1, 101, -1.0, -1});
    for (const auto & item : keep)
        if (item.second != std::vector<int>({62, 63})) return false;
    // Even an impossible mass cutoff or oversized global budget cannot remove top-2.
    keep = prune_policy::select(stats, {-1, -1, 1.0, -1});
    for (const auto & item : keep)
        if (item.second != std::vector<int>({62, 63})) return false;
    keep = prune_policy::select(stats, {-1, -1, -1.0, 1000});
    for (const auto & item : keep)
        if (item.second != std::vector<int>({62, 63})) return false;
    // Ties have a stable layer/expert ordering, making repeated mask sweeps reproducible.
    for (auto & item : stats) std::fill(item.second.msan.begin(), item.second.msan.end(), 1.0f);
    keep = prune_policy::select(stats, {-1, -1, -1.0, 1});
    if (keep.size() != 1 || keep.begin()->first != 1 || keep.begin()->second.size() != 63 ||
        keep.begin()->second.front() != 1 || keep.begin()->second.back() != 63) return false;
    keep = prune_policy::select(stats, {2, -1, -1.0, -1});
    for (const auto & item : keep)
        if (item.second != std::vector<int>({0, 1})) return false;
    return true;
}

int main() {
    if (!exercise()) { fprintf(stderr, "64-expert/top-2 prune policy failed\n"); return 1; }
    return 0;
}
