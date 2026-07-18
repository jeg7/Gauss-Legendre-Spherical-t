// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <coulomb.hcu>
#include <cuda_container.hcu>
#include <cuda_utils.hcu>
#include <glst_force.hcu>
#include <io.hpp>
#include <utils.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class glst_force_test_access {
public:
  static void enable_profiling(glst_force &force, const bool enabled) {
    force.profiling_enabled_ = enabled;
    return;
  }

  static const glst_profile &profile(const glst_force &force) {
    return force.profile_;
  }
};

int main(int argc, char **argv) {
  const bool cutoff_mode = ((argc == 11) || (argc == 12));
  const bool cell_mode = ((argc == 13) || (argc == 14));

  // Input check and error catch
  if ((!cutoff_mode) && (!cell_mode)) {
    std::cout << "Usage: " << argv[0]
              << " [system] [tol] [box_dim] [rcut] [G_cell] [G_tile] "
                 "[warmup_iterations] [benchmark_iterations] "
                 "[profile_iterations] [run_coulomb:0|1] [output_file|-]"
              << std::endl;
    std::cout << "OR" << std::endl;
    std::cout
        << "       " << argv[0]
        << " [sys] [tol] [box_dim] [ncell_x] [ncell_y] [ncell_z] [G_cell] "
           "[G_tile] [warmup_iterations] [benchmark_iterations] "
           "[profile_iterations] [run_coulomb:0|1] [output_file|-]"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::string file_name = "";
  double tol = 0.0;
  double box_dim_x = 0.0, box_dim_y = 0.0, box_dim_z = 0.0;
  double rcut = 0.0;

  unsigned int ncell_x = 0, ncell_y = 0, ncell_z = 0;

  unsigned int cell_partition_count = 0;
  unsigned int tile_partition_count = 0;
  unsigned int warmup_iterations = 0;
  unsigned int benchmark_iterations = 0;
  unsigned int profile_iterations = 0;

  bool run_coulomb = true;
  std::string output_path = "";

  try {
    int arg = 1;

    file_name = std::string(argv[arg++]);
    tol = parse_positive_double_arg(argv[arg++], "tol");

    box_dim_x = parse_positive_double_arg(argv[arg++], "box_dim");
    box_dim_y = box_dim_x;
    box_dim_z = box_dim_x;

    if (cutoff_mode)
      rcut = parse_positive_double_arg(argv[arg++], "rcut");
    else {
      ncell_x = parse_uint_arg(argv[arg++], "ncell_x", false);
      ncell_y = parse_uint_arg(argv[arg++], "ncell_y", false);
      ncell_z = parse_uint_arg(argv[arg++], "ncell_z", false);

      // Reproduce the existing ncell-based glst_force constructor exactly
      const double rcxd = box_dim_x / static_cast<double>(ncell_x);
      const double rcyd = box_dim_y / static_cast<double>(ncell_y);
      const double rczd = box_dim_z / static_cast<double>(ncell_z);

      rcut = rcxd;
      rcut = (rcyd < rcut) ? rcyd : rcut;
      rcut = (rczd < rcut) ? rczd : rcut;
    }

    cell_partition_count = parse_uint_arg(argv[arg++], "G_cell", false);
    tile_partition_count = parse_uint_arg(argv[arg++], "G_tile", false);

    warmup_iterations = parse_uint_arg(argv[arg++], "warmup_iterations", true);
    benchmark_iterations =
        parse_uint_arg(argv[arg++], "benchmark_iterations", false);
    profile_iterations =
        parse_uint_arg(argv[arg++], "profile_iterations", true);

    const std::string run_coulomb_argument(argv[arg++]);
    if ((run_coulomb_argument != "0") && (run_coulomb_argument != "1"))
      throw std::runtime_error("run_coulomb must be exactly 0 or 1");

    run_coulomb = (run_coulomb_argument == "1");

    if (arg < argc)
      output_path = std::string(argv[arg++]);

    if (arg != argc)
      throw std::runtime_error("Unexpected trailing command-line argument");
  } catch (const std::exception &error) {
    std::cerr << "Invalid command line: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::ofstream bench_out;
  std::streambuf *old_cout_buf = nullptr;

  if ((!output_path.empty()) && (output_path != "-")) {
    bench_out.open(output_path);

    if (!bench_out) {
      std::cerr << "Could not open output file: \"" << output_path << "\""
                << std::endl;
      return EXIT_FAILURE;
    }

    old_cout_buf = std::cout.rdbuf(bench_out.rdbuf());
  }

  int cuda_count = 0;
  cudaCheck(cudaGetDeviceCount(&cuda_count));

  const unsigned long long int requested_device_count =
      static_cast<unsigned long long int>(cell_partition_count) *
      static_cast<unsigned long long int>(tile_partition_count);

  if (requested_device_count !=
      static_cast<unsigned long long int>(cuda_count)) {
    std::cerr << "Invalid GPU layout: G_cell * G_tile must equal the visible "
                 "CUDA device count; observed "
              << cell_partition_count << " * " << tile_partition_count
              << " != " << cuda_count << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "BENCH_INPUT";
  std::cout << " sys=" << file_name;
  std::cout << " tol=" << tol;
  std::cout << " box_dim=" << box_dim_x << "," << box_dim_y << "," << box_dim_z;
  std::cout << " cut=" << rcut;

  if (cell_mode) {
    std::cout << " requested_cells=" << ncell_x << "," << ncell_y << ","
              << ncell_z;
  }

  std::cout << " visible_gpus=" << cuda_count;
  std::cout << " g_cell=" << cell_partition_count;
  std::cout << " g_tile=" << tile_partition_count;
  std::cout << " warmup_iterations=" << warmup_iterations;
  std::cout << " benchmark_iterations=" << benchmark_iterations;
  std::cout << " profile_iterations=" << profile_iterations;
  std::cout << " run_coulomb=" << (run_coulomb ? 1 : 0);
  std::cout << std::endl;

  // Allocate host memory and perform IO
  std::size_t natom = get_natom_psf(file_name + ".psf");
  cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);
  read_charmm_cor(rx.h_array(), ry.h_array(), rz.h_array(), natom,
                  file_name + ".cor");
  read_charmm_psf(qc.h_array(), natom, file_name + ".psf");
  recenter(rx.h_array(), ry.h_array(), rz.h_array(), natom);

  // Copy coordinates and charges to device
  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  // Compute GLST energy and forces
  cuda_container<double> fx_glst(natom), fy_glst(natom), fz_glst(natom),
      en_glst(natom);

  auto glst = std::make_unique<glst_force>();
  glst->set_gpu_layout(cell_partition_count, tile_partition_count);
  glst->init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);

  glst_force_test_access::enable_profiling(*glst, false);

  for (unsigned int iter = 0; iter < warmup_iterations; iter++) {
    glst->calc_ener_force(rx.d_array().data(), ry.d_array().data(),
                          rz.d_array().data(), qc.d_array().data());
  }

  std::vector<double> fused_times(benchmark_iterations);

  for (unsigned int iter = 0; iter < benchmark_iterations; iter++) {
    std::cerr << "\rBenchmark iteration " << iter + 1 << "/"
              << benchmark_iterations << std::flush;

    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();

    glst->calc_ener_force(rx.d_array().data(), ry.d_array().data(),
                          rz.d_array().data(), qc.d_array().data());

    const std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;

    fused_times[iter] = elapsed.count();
  }

  std::cerr << '\r' << std::string(80, ' ') << '\r';
  std::cout << "Finished " << benchmark_iterations << " benchmark calculations"
            << std::endl;
  std::cout << "Total GLST Runtime: " << avg(fused_times) << " ms (+/- "
            << stdev(fused_times) << " ms)" << std::endl;
  std::cout << std::endl;

  if (profile_iterations > 0) {
    std::vector<double> assignment_times(profile_iterations);
    std::vector<double> owned_halo_source_times(profile_iterations);
    std::vector<double> zero_ef_times(profile_iterations);

    std::vector<double> calc_sf_times(profile_iterations);
    std::vector<double> exchange_sf_times(profile_iterations);
    std::vector<double> sum_rmt_sf_times(profile_iterations);
    std::vector<double> calc_lr_ef_times(profile_iterations);

    std::vector<double> calc_sr_ef_times(profile_iterations);
    std::vector<double> reduce_tile_ef_times(profile_iterations);
    std::vector<double> instrumented_compute_times(profile_iterations);
    std::vector<double> gather_times(profile_iterations);

    // Derived timings. These are calculated per iteration so their standard
    // deviations include covariance between phases.
    std::vector<double> long_range_tile_times(profile_iterations);
    std::vector<double> long_range_glst_times(profile_iterations);
    std::vector<double> profiled_phase_sum_times(profile_iterations);
    std::vector<double> profiling_residual_times(profile_iterations);

    std::size_t sf_collective_input_bytes = 0;
    std::size_t owned_atom_replicas = 0;
    std::size_t halo_atom_replicas = 0;

    glst_force_test_access::enable_profiling(*glst, true);

    for (unsigned int iter = 0; iter < profile_iterations; iter++) {
      std::cerr << "\rProfiling iteration " << iter + 1 << "/"
                << profile_iterations << std::flush;

      glst->calc_ener_force(rx.d_array().data(), ry.d_array().data(),
                            rz.d_array().data(), qc.d_array().data());

      const glst_profile &profile = glst_force_test_access::profile(*glst);

      assignment_times[iter] = profile.atom_assignment_scatter_ms;
      owned_halo_source_times[iter] = profile.owned_halo_source_scatter_ms;
      zero_ef_times[iter] = profile.zero_ef_ms;
      calc_sf_times[iter] = profile.calc_sf_ms;
      exchange_sf_times[iter] = profile.exchange_sf_ms;
      sum_rmt_sf_times[iter] = profile.sum_rmt_sf_ms;
      calc_lr_ef_times[iter] = profile.calc_lr_ef_ms;
      calc_sr_ef_times[iter] = profile.calc_sr_ef_ms;
      reduce_tile_ef_times[iter] = profile.reduce_tile_ef_ms;
      instrumented_compute_times[iter] = profile.instrumented_compute_ms;

      // This is the tiled long-range loop itself:
      //
      // calc_sf_tile
      // exchange_sf_tile
      // sum_rmt_sf_tile
      // calc_lr_ef_tile
      long_range_tile_times[iter] =
          calc_sf_times[iter] + exchange_sf_times[iter] +
          sum_rmt_sf_times[iter] + calc_lr_ef_times[iter];

      // Preserve the scope of the previous "Long-Range GLST Runtime":
      // Assignment/preparation plus all long-range work, before short range.
      long_range_glst_times[iter] =
          assignment_times[iter] + owned_halo_source_times[iter] +
          zero_ef_times[iter] + long_range_tile_times[iter];

      profiled_phase_sum_times[iter] = long_range_glst_times[iter] +
                                       calc_sr_ef_times[iter] +
                                       reduce_tile_ef_times[iter];

      profiling_residual_times[iter] =
          instrumented_compute_times[iter] - profiled_phase_sum_times[iter];

      // get_ef() is outside calc_ener_force(), so time the final
      // gather/reordering separately.
      const std::chrono::steady_clock::time_point gather_start =
          std::chrono::steady_clock::now();

      glst->get_ef(fx_glst, fy_glst, fz_glst, en_glst);

      const std::chrono::steady_clock::time_point gather_end =
          std::chrono::steady_clock::now();

      const std::chrono::duration<double, std::milli> gather_elapsed =
          gather_end - gather_start;

      gather_times[iter] = gather_elapsed.count();

      // These are topology/data-volume counters, not timings. They should
      // remain constant because full_example repeats the same coordinates.
      if (iter == 0) {
        sf_collective_input_bytes = profile.sf_collective_input_bytes;
        owned_atom_replicas = profile.owned_atom_replicas;
        halo_atom_replicas = profile.halo_atom_replicas;
      } else if ((sf_collective_input_bytes !=
                  profile.sf_collective_input_bytes) ||
                 (owned_atom_replicas != profile.owned_atom_replicas) ||
                 (halo_atom_replicas != profile.halo_atom_replicas)) {
        throw std::runtime_error(
            "Profiling counters changed between identical calculations");
      }
    }

    if (tile_partition_count > 1) {
      std::cout
          << "Profiling phase timings serialize tile partitions; use the "
             "non-instrumented Total GLST Runtime for end-to-end performance."
          << std::endl;
      std::cout << std::endl;
    }

    std::cerr << '\r' << std::string(80, ' ') << '\r';
    std::cout << "Finished " << profile_iterations << " profiling calculations"
              << std::endl;
    std::cout << std::endl;

    std::cout << "           Classify and assign owned atoms: "
              << avg(assignment_times) << " ms (+/- " << stdev(assignment_times)
              << " ms)" << std::endl;
    std::cout << " Construct/scatter owned+halo atom sources: "
              << avg(owned_halo_source_times) << " ms (+/- "
              << stdev(owned_halo_source_times) << " ms)" << std::endl;
    std::cout << "                  Zero energy/force arrays: "
              << avg(zero_ef_times) << " ms (+/- " << stdev(zero_ef_times)
              << " ms)" << std::endl;
    std::cout << "               Calculate structure factors: "
              << avg(calc_sf_times) << " ms (+/- " << stdev(calc_sf_times)
              << " ms)" << std::endl;
    std::cout << "           Exchange structure-factor tiles: "
              << avg(exchange_sf_times) << " ms (+/- "
              << stdev(exchange_sf_times) << " ms)" << std::endl;
    std::cout << "              Sum remote structure factors: "
              << avg(sum_rmt_sf_times) << " ms (+/- " << stdev(sum_rmt_sf_times)
              << " ms)" << std::endl;
    std::cout << "    Calculate long-range energy and forces: "
              << avg(calc_lr_ef_times) << " ms (+/- " << stdev(calc_lr_ef_times)
              << " ms)" << std::endl;
    std::cout << "   Calculate short-range energy and forces: "
              << avg(calc_sr_ef_times) << " ms (+/- " << stdev(calc_sr_ef_times)
              << " ms)" << std::endl;
    std::cout << "       Reduce tile-partition energy/forces: "
              << avg(reduce_tile_ef_times) << " ms (+/- "
              << stdev(reduce_tile_ef_times) << " ms)" << std::endl;
    std::cout << "                Gather and reorder results: "
              << avg(gather_times) << " ms (+/- " << stdev(gather_times)
              << " ms)" << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "-------------"
              << std::endl;
    std::cout << "              Long-range tile-loop runtime: "
              << avg(long_range_tile_times) << " ms (+/- "
              << stdev(long_range_tile_times) << " ms)" << std::endl;
    std::cout << "                   Long-Range GLST Runtime: "
              << avg(long_range_glst_times) << " ms (+/- "
              << stdev(long_range_glst_times) << " ms)" << std::endl;
    std::cout << "                   Profiled phase-sum time: "
              << avg(profiled_phase_sum_times) << " ms (+/- "
              << stdev(profiled_phase_sum_times) << " ms)" << std::endl;
    std::cout << "                    Instrumented GLST time: "
              << avg(instrumented_compute_times) << " ms (+/- "
              << stdev(instrumented_compute_times) << " ms)" << std::endl;
    std::cout << "                   Profiling time residual: "
              << avg(profiling_residual_times) << " ms (+/- "
              << stdev(profiling_residual_times) << " ms)" << std::endl;

    const double sf_collective_input_mib =
        static_cast<double>(sf_collective_input_bytes) / (1024.0 * 1024.0);
    const std::size_t total_atom_replicas =
        owned_atom_replicas + halo_atom_replicas;

    std::cout << std::endl;
    std::cout << " Aggregate logical S_tile collective input: "
              << sf_collective_input_mib << " MiB ("
              << sf_collective_input_bytes << " bytes)" << std::endl;
    std::cout << "             Owned atom replicas: " << owned_atom_replicas
              << std::endl;
    std::cout << "              Halo atom replicas: " << halo_atom_replicas
              << std::endl;
    std::cout << "             Total atom replicas: " << total_atom_replicas
              << std::endl;
  }

  // The final profiling iteration already called get_ef(). When profiling is
  // disabled gather the result from the final production benchmark instead.
  if (profile_iterations == 0)
    glst->get_ef(fx_glst, fy_glst, fz_glst, en_glst);

  if (run_coulomb) {
    // Compute Coulomb energy and forces
    cudaSetDevice(0);
    cuda_container<double> fx_coul(natom), fy_coul(natom), fz_coul(natom),
        en_coul(natom);

    auto start_coul = std::chrono::steady_clock::now();
    compute_coulomb_cuda(fx_coul.d_array().data(), fy_coul.d_array().data(),
                         fz_coul.d_array().data(), en_coul.d_array().data(),
                         rx.d_array().data(), ry.d_array().data(),
                         rz.d_array().data(), qc.d_array().data(), natom);
    auto end_coul = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_coul = end_coul - start_coul;
    std::cout << std::endl;
    std::cout << "Coulomb Runtime: " << time_coul.count() << " ms" << std::endl;

    // Transfer Coulomb results to host
    cudaSetDevice(0);
    fx_coul.transfer_to_host();
    fy_coul.transfer_to_host();
    fz_coul.transfer_to_host();
    en_coul.transfer_to_host();

    // Print and compute errors
    print_error_report(fx_glst.h_array(), fy_glst.h_array(), fz_glst.h_array(),
                       en_glst.h_array(), fx_coul.h_array(), fy_coul.h_array(),
                       fz_coul.h_array(), en_coul.h_array(), natom, tol);
  }

  if (old_cout_buf != nullptr)
    std::cout.rdbuf(old_cout_buf);

  return 0;
}
