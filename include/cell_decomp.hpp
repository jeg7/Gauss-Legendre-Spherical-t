// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __GLST_CELL_DECOMP__
#define __GLST_CELL_DECOMP__

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

struct Range1D {
  int begin = 0; // Inclusive
  int end = -1;  // Inclusive
  int size(void) const { return ((end >= begin) ? (end - begin + 1) : 0); }
};

struct ProcGrid {
  int Px = 1, Py = 1, Pz = 1;
};

struct CellDistribution {
  int ncell_x = 0, ncell_y = 0, ncell_z = 0;
  int ngpu_total = 0;
  int ngpu_used = 0;
  ProcGrid grid;

  std::vector<int> cell_dev_idx;
  std::vector<std::vector<unsigned int>> dev_cell_idx;
};

// Split N items into P contiguous parts, sizes differ by at most 1.
// Part p gets [begin,end] inclusive.
inline Range1D split_range_1d(const int N, const int P, const int p) {
  if (P <= 0)
    throw std::runtime_error("split_range_1d: P must be > 0");
  if ((p < 0) || (p >= P))
    throw std::runtime_error("split_range_1d: p out of range");
  if (N < 0)
    throw std::runtime_error("split_range_1d: N must be >= 0");

  const int base = (P == 0) ? 0 : (N / P);
  const int rem = (P == 0) ? 0 : (N % P);

  const int sz = base + ((p < rem) ? 1 : 0); // Number of elements in range
  const int b = p * base + std::min(p, rem); // Beginning of range
  const int e = b + sz - 1;                  // End of range

  return Range1D{b, e};
}

inline int num_decomp_dims(const ProcGrid &g) {
  return (g.Px > 1) + (g.Py > 1) + (g.Pz > 1);
}

inline int rank_to_dev(const int px, const int py, const int pz,
                       const ProcGrid &g) {
  return (px * g.Py + py) * g.Pz + pz;
}

inline unsigned int cell_to_idx(const int x, const int y, const int z,
                                const int ncell_y, const int ncell_z) {
  return static_cast<unsigned int>((x * ncell_y + y) * ncell_z + z);
}

inline ProcGrid choose_best_proc_grid(const int nx, const int ny, const int nz,
                                      const int ngpu) {
  if ((nx <= 0) || (ny <= 0) || (nz <= 0))
    throw std::runtime_error("choose_best_proc_grid: nx, ny, nz must be > 0");
  if (ngpu <= 0)
    throw std::runtime_error("choose_best_proc_grid: ngpu must be > 0");

  bool found = false;
  ProcGrid best;
  int best_ndim = std::numeric_limits<int>::max();
  double best_aspect = std::numeric_limits<double>::infinity();
  int best_max_vol = std::numeric_limits<int>::max();

  auto eval_candidate = [&](const ProcGrid &g) -> void {
    // Compute exact worst-case aspect ratio and volume across all ranks
    double max_aspect = 0.0;
    int max_vol = 0;

    for (int px = 0; px < g.Px; px++) {
      const Range1D xr = split_range_1d(nx, g.Px, px);
      const int sx = xr.size();
      for (int py = 0; py < g.Py; py++) {
        const Range1D yr = split_range_1d(ny, g.Py, py);
        const int sy = yr.size();
        for (int pz = 0; pz < g.Pz; pz++) {
          const Range1D zr = split_range_1d(nz, g.Pz, pz);
          const int sz = zr.size();

          const int mn = std::min(sx, std::min(sy, sz));
          const int mx = std::max(sx, std::max(sy, sz));
          const double aspect =
              (mn > 0) ? (static_cast<double>(mx) / static_cast<double>(mn))
                       : std::numeric_limits<double>::infinity();

          max_aspect = std::max(max_aspect, aspect);

          const int vol = sx * sy * sz;
          max_vol = std::max(max_vol, vol);
        }
      }
    }

    const int ndim = num_decomp_dims(g);

    // Lexicographic "better than" check using the rule above
    const bool better =
        (!found) || (ndim < best_ndim) ||
        ((ndim == best_ndim) && (max_aspect < best_aspect)) ||
        ((ndim == best_ndim) && (max_aspect == best_aspect) &&
         (max_vol < best_max_vol)) ||
        ((ndim == best_ndim) && (max_aspect == best_aspect) &&
         (max_vol == best_max_vol) &&
         (std::tie(g.Px, g.Py, g.Pz) > std::tie(best.Px, best.Py, best.Pz)));

    if (better) {
      found = true;
      best = g;
      best_ndim = ndim;
      best_aspect = max_aspect;
      best_max_vol = max_vol;
    }
  };

  // Enumerate all factor triples
  for (int Px = 1; Px <= ngpu; Px++) {
    if (ngpu % Px != 0)
      continue;
    const int rem1 = ngpu / Px;
    for (int Py = 1; Py <= rem1; Py++) {
      if (rem1 % Py != 0)
        continue;
      const int Pz = rem1 / Py;

      ProcGrid g{Px, Py, Pz};
      if ((g.Px <= nx) && (g.Py <= ny) && (g.Pz <= nz))
        eval_candidate(g);
    }
  }

  if (!found) {
    throw std::runtime_error(
        "choose_best_proc_grid: No valid brick decomposition exists for ngpu "
        "with these grid sizes.\nHint: this can happen if ngpu has a factor "
        "larger than nx, ny, nz.\nYou can reduce ngpu_used or use a non-brick "
        "fallback (e.g. linear chunking).");
  }

  return best;
}

inline CellDistribution distribute_cells_to_gpus(const int ncell_x,
                                                 const int ncell_y,
                                                 const int ncell_z,
                                                 const int ngpu_total) {
  if ((ncell_z <= 0) || (ncell_y <= 0) || (ncell_z <= 0)) {
    throw std::runtime_error(
        "distribute_cells_to_gpus: ncell dims must be > 0");
  }
  if (ngpu_total <= 0) {
    throw std::runtime_error(
        "distribute_cells_to_gpus: ngpu_total must be > 0");
  }

  const std::int64_t ncell_total64 = static_cast<std::int64_t>(ncell_x) *
                                     static_cast<std::int64_t>(ncell_y) *
                                     static_cast<std::int64_t>(ncell_z);
  if (ncell_total64 > std::numeric_limits<int>::max()) {
    throw std::runtime_error(
        "distribute_cells_to_gpus: ncell_total too large for int indexing");
  }
  const int ncell_total = static_cast<int>(ncell_total64);

  CellDistribution dist;
  dist.ncell_x = ncell_x;
  dist.ncell_y = ncell_y;
  dist.ncell_z = ncell_z;
  dist.ngpu_total = ngpu_total;

  // You cannot give every GPU at least one cell if ncell_total < ngpu_total.
  // In that case, ngpu_used is limited by ncell_total
  dist.ngpu_used = std::min(ngpu_total, ncell_total);

  // Choose a process grid for ngpu_used GPUs
  // If this throws (rare for power-of-two GPU counts on reasonable grids), you
  // can catch and implement a fallback
  dist.grid = choose_best_proc_grid(ncell_x, ncell_y, ncell_z, dist.ngpu_used);

  dist.cell_dev_idx.assign(ncell_total, -1);
  dist.dev_cell_idx.assign(ngpu_total, {});

  // Precompute per-dimension ranges
  std::vector<Range1D> xr(dist.grid.Px), yr(dist.grid.Py), zr(dist.grid.Pz);
  for (int px = 0; px < dist.grid.Px; px++)
    xr[px] = split_range_1d(ncell_x, dist.grid.Px, px);
  for (int py = 0; py < dist.grid.Py; py++)
    yr[py] = split_range_1d(ncell_y, dist.grid.Py, py);
  for (int pz = 0; pz < dist.grid.Pz; pz++)
    zr[pz] = split_range_1d(ncell_z, dist.grid.Pz, pz);

  // Assign cells to GPUs by continguous bricks
  for (int px = 0; px < dist.grid.Px; px++) {
    for (int py = 0; py < dist.grid.Py; py++) {
      for (int pz = 0; pz < dist.grid.Pz; pz++) {
        const int dev = rank_to_dev(px, py, pz, dist.grid);
        const Range1D X = xr[px];
        const Range1D Y = yr[py];
        const Range1D Z = zr[pz];

        auto &cell_list = dist.dev_cell_idx[dev];
        cell_list.reserve(X.size() * Y.size() * Z.size());

        for (int x = X.begin; x <= X.end; x++) {
          for (int y = Y.begin; y <= Y.end; y++) {
            for (int z = Z.begin; z <= Z.end; z++) {
              const unsigned int cell = cell_to_idx(x, y, z, ncell_y, ncell_z);
              dist.cell_dev_idx[cell] = dev;
              cell_list.push_back(cell);
            }
          }
        }
      }
    }
  }

  // Sanity check: all cells assigned?
  for (int cell = 0; cell < ncell_total; cell++) {
    if (dist.cell_dev_idx[cell] < 0) {
      throw std::runtime_error(
          "distribute_cells_to_gpus: internal error: unassigned cell");
    }
  }

  return dist;
}

#endif
