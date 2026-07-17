// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <cuda_container.hcu>
#include <cuda_utils.hcu>
#include <glst_force.hcu>
#include <glst_plan.hcu>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/*
 * This executable intentionally accepts no command-line arguments. The parent
 * process launches private child instances of itself with internal environment
 * variables so the 1 x 1 reference can run with one visible GPU while the
 * comparison layouts run with every GPU visible to the original process.
 */

class glst_force_test_access {
public:
  static const glst_plan &plan(const glst_force &force) {
    if (force.plan_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::plan: Plan is not initialized");
    }

    return *(force.plan_);
  }
};

struct test_case {
  std::string selector;
  std::string name;
  unsigned int natom;
  double tol;
  double box;
  double rcut;
  unsigned int expected_ncell_axis;
  unsigned int expected_ngroup;
  unsigned int expected_total_nodes;
  std::vector<unsigned int> expected_group_nodes;
};

struct gpu_layout {
  unsigned int cell_partition_count;
  unsigned int tile_partition_count;
};

struct ef_result {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  std::vector<double> en;
};

struct error_stats {
  double average_abs;
  double rmse;
  double max_abs;
  std::size_t max_abs_index;
  double worst_tolerance_ratio;
  std::size_t worst_tolerance_index;
  double worst_observed;
  double worst_reference;
  double worst_abs_error;
  double worst_allowed;
};

struct reference_header {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t natom;
  std::uint32_t ncell_x;
  std::uint32_t ncell_y;
  std::uint32_t ncell_z;
  std::uint32_t ngroup;
  std::uint32_t total_nodes;
  std::uint32_t max_tile_nodes;
  std::uint32_t tile_count;
  double tol;
  double box;
  double rcut;
};

constexpr std::uint64_t reference_magic = 0x474c535448594246ULL;
constexpr std::uint32_t reference_version = 1u;
constexpr double absolute_tolerance = 1.0e-10;
constexpr double relative_tolerance = 1.0e-9;

constexpr const char *stage_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_STAGE";
constexpr const char *probe_path_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_PROBE_PATH";
constexpr const char *case_environment = "GLST_HYBRID_LAYOUT_CORRECTNESS_CASE";
constexpr const char *cell_partition_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_G_CELL";
constexpr const char *tile_partition_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_G_TILE";
constexpr const char *reference_mode_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_REFERENCE_MODE";
constexpr const char *reference_path_environment =
    "GLST_HYBRID_LAYOUT_CORRECTNESS_REFERENCE_PATH";

static_assert(sizeof(double) == sizeof(std::uint64_t),
              "The reference format requires 64-bit double values");

class temporary_file {
public:
  explicit temporary_file(const std::string &label) : path_() {
    std::string pattern = "/tmp/glst_hybrid_layout_";
    pattern += std::to_string(static_cast<long long int>(getpid()));
    pattern += "_" + label + "_XXXXXX";

    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');

    const int descriptor = mkstemp(mutable_pattern.data());
    if (descriptor < 0) {
      throw std::runtime_error("Could not create temporary file for " + label +
                               ": " + std::strerror(errno));
    }

    if (close(descriptor) != 0) {
      const int close_error = errno;
      unlink(mutable_pattern.data());
      throw std::runtime_error("Could not close temporary file for " + label +
                               ": " + std::strerror(close_error));
    }

    this->path_ = std::string(mutable_pattern.data());
  }

  temporary_file(const temporary_file &) = delete;
  temporary_file &operator=(const temporary_file &) = delete;

  ~temporary_file(void) {
    if (!this->path_.empty())
      unlink(this->path_.c_str());
  }

  const std::string &path(void) const { return this->path_; }

private:
  std::string path_;
};

static void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

static std::vector<test_case> test_cases(void) {
  std::vector<test_case> cases;

  cases.push_back({"small", "small_2959", 2959u, 1.0e-6, 32.0, 12.0, 3u, 1u, 0u,
                   std::vector<unsigned int>()});

  cases.push_back({"large", "sample_197159", 197159u, 1.0e-6, 128.0, 12.0, 11u,
                   4u, 95148u,
                   std::vector<unsigned int>{52173u, 25569u, 12894u, 4512u}});

  return cases;
}

static std::string environment_value(const char *name) {
  const char *value = std::getenv(name);

  if ((value == nullptr) || (value[0] == '\0'))
    throw std::runtime_error(
        std::string("Required internal environment value is ") +
        "missing: " + name);

  return std::string(value);
}

static unsigned int parse_positive_uint(const std::string &text,
                                        const std::string &label) {
  require(!text.empty(), label + " is empty");

  for (std::size_t i = 0; i < text.size(); i++) {
    if ((text[i] < '0') || (text[i] > '9'))
      throw std::runtime_error(label + " is not a positive integer: " + text);
  }

  std::size_t consumed = 0;
  unsigned long long int parsed = 0;

  try {
    parsed = std::stoull(text, &consumed, 10);
  } catch (const std::exception &) {
    throw std::runtime_error(
        label + " is outside the supported integer range: " + text);
  }

  require(consumed == text.size(),
          label + " contains trailing characters: " + text);
  require(parsed > 0, label + " must be greater than zero");
  require(parsed <= static_cast<unsigned long long int>(
                        std::numeric_limits<unsigned int>::max()),
          label + " exceeds unsigned int range");

  return static_cast<unsigned int>(parsed);
}

static std::string trim(const std::string &text) {
  std::size_t first = 0;
  while ((first < text.size()) &&
         ((text[first] == ' ') || (text[first] == '\t') ||
          (text[first] == '\n') || (text[first] == '\r'))) {
    first++;
  }

  std::size_t last = text.size();
  while ((last > first) &&
         ((text[last - 1] == ' ') || (text[last - 1] == '\t') ||
          (text[last - 1] == '\n') || (text[last - 1] == '\r'))) {
    last--;
  }

  return text.substr(first, last - first);
}

static std::string executable_path(void) {
  std::vector<char> path_buffer(4096, '\0');

  while (true) {
    const ssize_t path_length =
        readlink("/proc/self/exe", path_buffer.data(), path_buffer.size() - 1);

    if (path_length < 0) {
      throw std::runtime_error("Could not resolve /proc/self/exe: " +
                               std::string(std::strerror(errno)));
    }

    if (static_cast<std::size_t>(path_length) < path_buffer.size() - 1) {
      return std::string(path_buffer.data(),
                         static_cast<std::size_t>(path_length));
    }

    require(path_buffer.size() <= std::numeric_limits<std::size_t>::max() / 2,
            "Executable path buffer size overflow");
    path_buffer.resize(path_buffer.size() * 2, '\0');
  }
}

static int run_child_process(
    const std::string &program,
    const std::vector<std::pair<std::string, std::string>> &environment) {
  std::cout.flush();
  std::cerr.flush();

  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error("Could not fork test child: " +
                             std::string(std::strerror(errno)));
  }

  if (child == 0) {
    for (std::size_t i = 0; i < environment.size(); i++) {
      const std::string &name = environment[i].first;
      const std::string &value = environment[i].second;

      if (setenv(name.c_str(), value.c_str(), 1) != 0) {
        std::fprintf(stderr, "Could not set child environment %s: %s\n",
                     name.c_str(), std::strerror(errno));
        _exit(127);
      }
    }

    char *const child_arguments[] = {const_cast<char *>(program.c_str()),
                                     nullptr};

    execv(program.c_str(), child_arguments);

    std::fprintf(stderr, "Could not execute test child %s: %s\n",
                 program.c_str(), std::strerror(errno));
    _exit(127);
  }

  int child_status = 0;
  pid_t waited_child = -1;

  do {
    waited_child = waitpid(child, &child_status, 0);
  } while ((waited_child < 0) && (errno == EINTR));

  if (waited_child < 0) {
    throw std::runtime_error("Could not wait for test child: " +
                             std::string(std::strerror(errno)));
  }

  if (WIFSIGNALED(child_status)) {
    std::ostringstream message;
    message << "Test child terminated by signal " << WTERMSIG(child_status);
    throw std::runtime_error(message.str());
  }

  require(WIFEXITED(child_status), "Test child did not exit normally");

  return WEXITSTATUS(child_status);
}

static void require_child_success(
    const std::string &program,
    const std::vector<std::pair<std::string, std::string>> &environment,
    const std::string &description) {
  const int status = run_child_process(program, environment);

  if (status != EXIT_SUCCESS) {
    std::ostringstream message;
    message << description << " failed with exit status " << status;
    throw std::runtime_error(message.str());
  }
}

static std::vector<std::string>
visible_device_tokens(const int visible_cuda_count) {
  require(visible_cuda_count > 0, "Visible CUDA device count must be positive");

  const char *visible_environment = std::getenv("CUDA_VISIBLE_DEVICES");
  std::vector<std::string> tokens;

  if (visible_environment != nullptr) {
    const std::string visible_text(visible_environment);
    std::size_t point = 0;

    while (point <= visible_text.size()) {
      const std::size_t comma = visible_text.find(',', point);
      const std::size_t count =
          (comma == std::string::npos) ? std::string::npos : comma - point;
      const std::string token = trim(visible_text.substr(point, count));

      if (!token.empty())
        tokens.push_back(token);

      if (comma == std::string::npos)
        break;

      point = comma + 1;
    }
  }

  if (tokens.empty()) {
    for (int dev = 0; dev < visible_cuda_count; dev++)
      tokens.push_back(std::to_string(dev));
  }

  require(tokens.size() >= static_cast<std::size_t>(visible_cuda_count),
          "CUDA_VISIBLE_DEVICES contains fewer entries than cudaGetDeviceCount "
          "reported");

  tokens.resize(static_cast<std::size_t>(visible_cuda_count));
  return tokens;
}

static std::string join_device_tokens(const std::vector<std::string> &tokens,
                                      const std::size_t token_count) {
  require(token_count > 0, "Requested CUDA device subset is empty");
  require(token_count <= tokens.size(),
          "Requested CUDA device subset exceeds available tokens");

  std::string joined;

  for (std::size_t i = 0; i < token_count; i++) {
    if (i > 0)
      joined += ",";

    joined += tokens[i];
  }

  return joined;
}

static std::vector<gpu_layout>
multiple_gpu_layouts(const unsigned int visible_cuda_count) {
  std::vector<gpu_layout> layouts;

  if (visible_cuda_count <= 1u)
    return layouts;

  layouts.push_back({visible_cuda_count, 1u});
  layouts.push_back({1u, visible_cuda_count});

  for (unsigned int cell_partition_count = 2u;
       cell_partition_count < visible_cuda_count; cell_partition_count++) {
    if ((visible_cuda_count % cell_partition_count) != 0u)
      continue;

    const unsigned int tile_partition_count =
        visible_cuda_count / cell_partition_count;

    if (tile_partition_count <= 1u)
      continue;

    layouts.push_back({cell_partition_count, tile_partition_count});
  }

  return layouts;
}

template <typename T>
static void write_scalar(std::ostream &output, const T &value,
                         const std::string &label) {
  output.write(reinterpret_cast<const char *>(&value),
               static_cast<std::streamsize>(sizeof(T)));

  if (!output)
    throw std::runtime_error("Could not write " + label);
}

template <typename T>
static void read_scalar(std::istream &input, T &value,
                        const std::string &label) {
  input.read(reinterpret_cast<char *>(&value),
             static_cast<std::streamsize>(sizeof(T)));

  if (!input)
    throw std::runtime_error("Could not read " + label);
}

template <typename T>
static void write_vector(std::ostream &output, const std::vector<T> &values,
                         const std::string &label) {
  if (values.empty())
    return;

  require(values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T),
          label + " byte count overflows size_t");

  const std::size_t byte_count = values.size() * sizeof(T);

  require(byte_count <= static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max()),
          label + " byte count exceeds streamsize range");

  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(byte_count));

  if (!output)
    throw std::runtime_error("Could not write " + label);
}

template <typename T>
static void read_vector(std::istream &input, std::vector<T> &values,
                        const std::string &label) {
  if (values.empty())
    return;

  require(values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T),
          label + " byte count overflows size_t");

  const std::size_t byte_count = values.size() * sizeof(T);

  require(byte_count <= static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max()),
          label + " byte count exceeds streamsize range");

  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(byte_count));

  if (!input)
    throw std::runtime_error("Could not read " + label);
}

static ef_result run_calculation(glst_force &force,
                                 const cuda_container<double> &rx,
                                 const cuda_container<double> &ry,
                                 const cuda_container<double> &rz,
                                 const cuda_container<double> &qc,
                                 const unsigned int natom) {
  force.calc_ener_force(rx.d_array().data(), ry.d_array().data(),
                        rz.d_array().data(), qc.d_array().data());

  cuda_container<double> fx;
  cuda_container<double> fy;
  cuda_container<double> fz;
  cuda_container<double> en;

  force.get_ef(fx, fy, fz, en);

  const std::size_t expected_size = static_cast<std::size_t>(natom);

  require(fx.size() == expected_size,
          "get_ef returned an unexpected x-force count");
  require(fy.size() == expected_size,
          "get_ef returned an unexpected y-force count");
  require(fz.size() == expected_size,
          "get_ef returned an unexpected z-force count");
  require(en.size() == expected_size,
          "get_ef returned an unexpected energy count");

  ef_result result;
  result.fx = fx.h_array();
  result.fy = fy.h_array();
  result.fz = fz.h_array();
  result.en = en.h_array();

  return result;
}

static void validate_component(const std::vector<double> &values,
                               const std::size_t expected_size,
                               const std::string &label) {
  require(values.size() == expected_size,
          label + " contains an unexpected number of values");

  for (std::size_t atom = 0; atom < values.size(); atom++) {
    if (!std::isfinite(values[atom])) {
      std::ostringstream message;
      message << label << " contains a non-finite value at atom " << atom
              << ": " << values[atom];
      throw std::runtime_error(message.str());
    }
  }
}

static void verify_repeat_component(const std::vector<double> &first,
                                    const std::vector<double> &second,
                                    const std::string &component,
                                    const std::string &case_name,
                                    const unsigned int cell_partition_count,
                                    const unsigned int tile_partition_count) {
  require(first.size() == second.size(),
          component + " repeat arrays have different sizes");

  for (std::size_t atom = 0; atom < first.size(); atom++) {
    if (std::memcmp(static_cast<const void *>(&first[atom]),
                    static_cast<const void *>(&second[atom]),
                    sizeof(double)) == 0) {
      continue;
    }

    std::uint64_t first_bits = 0;
    std::uint64_t second_bits = 0;
    std::memcpy(static_cast<void *>(&first_bits),
                static_cast<const void *>(&first[atom]), sizeof(double));
    std::memcpy(static_cast<void *>(&second_bits),
                static_cast<const void *>(&second[atom]), sizeof(double));

    std::ostringstream message;
    message << std::setprecision(17)
            << "Deterministic repeatability failure: case=" << case_name
            << ", layout=" << cell_partition_count << "x"
            << tile_partition_count << ", component=" << component
            << ", atom=" << atom << ", first=" << first[atom]
            << ", second=" << second[atom]
            << ", abs_difference=" << std::abs(first[atom] - second[atom])
            << ", first_bits=" << std::hex << std::showbase << first_bits
            << ", second_bits=" << second_bits;
    throw std::runtime_error(message.str());
  }
}

static error_stats calculate_error_stats(const std::vector<double> &observed,
                                         const std::vector<double> &reference,
                                         const std::string &component) {
  require(!observed.empty(), component + " comparison contains no values");
  require(observed.size() == reference.size(),
          component + " comparison arrays have different sizes");

  long double sum_abs = 0.0L;
  long double sum_squared = 0.0L;

  error_stats stats = {0.0, 0.0, 0.0, 0u, 0.0, 0u, 0.0, 0.0, 0.0, 0.0};

  for (std::size_t atom = 0; atom < observed.size(); atom++) {
    if (!std::isfinite(observed[atom])) {
      std::ostringstream message;
      message << component << " observed value is not finite at atom " << atom;
      throw std::runtime_error(message.str());
    }

    if (!std::isfinite(reference[atom])) {
      std::ostringstream message;
      message << component << " reference value is not finite at atom " << atom;
      throw std::runtime_error(message.str());
    }

    const double difference = observed[atom] - reference[atom];
    const double abs_error = std::abs(difference);
    const double scale =
        std::max(std::abs(observed[atom]), std::abs(reference[atom]));
    const double allowed = absolute_tolerance + relative_tolerance * scale;
    const double ratio = abs_error / allowed;

    sum_abs += static_cast<long double>(abs_error);
    sum_squared += static_cast<long double>(difference) *
                   static_cast<long double>(difference);

    if ((atom == 0) || (abs_error > stats.max_abs)) {
      stats.max_abs = abs_error;
      stats.max_abs_index = atom;
    }

    if ((atom == 0) || (ratio > stats.worst_tolerance_ratio)) {
      stats.worst_tolerance_ratio = ratio;
      stats.worst_tolerance_index = atom;
      stats.worst_observed = observed[atom];
      stats.worst_reference = reference[atom];
      stats.worst_abs_error = abs_error;
      stats.worst_allowed = allowed;
    }
  }

  const long double count = static_cast<long double>(observed.size());
  stats.average_abs = static_cast<double>(sum_abs / count);
  stats.rmse = static_cast<double>(std::sqrt(sum_squared / count));

  return stats;
}

static void print_error_stats(const std::string &component,
                              const error_stats &stats) {
  std::cout << "  " << component << ": average_abs=" << std::scientific
            << std::setprecision(6) << stats.average_abs
            << ", rmse=" << stats.rmse << ", max_abs=" << stats.max_abs
            << ", max_index=" << stats.max_abs_index
            << ", worst_tolerance_ratio=" << stats.worst_tolerance_ratio
            << ", worst_tolerance_index=" << stats.worst_tolerance_index
            << std::endl;
}

static void enforce_component_tolerance(
    const std::string &component, const error_stats &stats,
    const std::string &case_name, const unsigned int cell_partition_count,
    const unsigned int tile_partition_count) {
  if (stats.worst_tolerance_ratio <= 1.0)
    return;

  std::ostringstream message;
  message << std::setprecision(17)
          << "Layout comparison failure: case=" << case_name
          << ", layout=" << cell_partition_count << "x" << tile_partition_count
          << ", component=" << component
          << ", atom=" << stats.worst_tolerance_index
          << ", observed=" << stats.worst_observed
          << ", reference=" << stats.worst_reference
          << ", abs_error=" << stats.worst_abs_error
          << ", allowed=" << stats.worst_allowed
          << ", ratio=" << stats.worst_tolerance_ratio;
  throw std::runtime_error(message.str());
}

static int run_probe_child(void) {
  try {
    const std::string probe_path = environment_value(probe_path_environment);

    int visible_cuda_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&visible_cuda_count);

    if (status != cudaSuccess) {
      throw std::runtime_error(
          "cudaGetDeviceCount failed while probing visible GPUs: " +
          std::string(cudaGetErrorString(status)));
    }

    std::ofstream output(probe_path, std::ios::trunc);
    require(static_cast<bool>(output),
            "Could not open CUDA-device probe file for writing");

    output << visible_cuda_count << std::endl;
    output.close();

    require(static_cast<bool>(output),
            "Could not finish writing CUDA-device probe file");
  } catch (const std::exception &e) {
    std::cerr << "FAIL hybrid_layout_correctness probe: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int run_layout_child(void) {
  try {
    const std::string case_selector = environment_value(case_environment);
    const unsigned int cell_partition_count = parse_positive_uint(
        environment_value(cell_partition_environment), "G_cell");
    const unsigned int tile_partition_count = parse_positive_uint(
        environment_value(tile_partition_environment), "G_tile");
    const std::string reference_mode =
        environment_value(reference_mode_environment);
    const std::string reference_path =
        environment_value(reference_path_environment);

    require((reference_mode == "write") || (reference_mode == "compare"),
            "Internal reference mode must be write or compare");

    const std::vector<test_case> cases = test_cases();
    const test_case *selected_case = nullptr;

    for (std::size_t i = 0; i < cases.size(); i++) {
      if (cases[i].selector == case_selector) {
        selected_case = &cases[i];
        break;
      }
    }

    require(selected_case != nullptr,
            "Unknown internal test case selector: " + case_selector);

    const test_case &current_case = *selected_case;

    int visible_cuda_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&visible_cuda_count);

    if (count_status != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceCount failed: " +
                               std::string(cudaGetErrorString(count_status)));
    }

    require(visible_cuda_count > 0, "No CUDA-capable device is visible");

    const unsigned long long int layout_product =
        static_cast<unsigned long long int>(cell_partition_count) *
        static_cast<unsigned long long int>(tile_partition_count);

    require(layout_product ==
                static_cast<unsigned long long int>(visible_cuda_count),
            "G_cell * G_tile does not equal the visible CUDA device count");

    if (reference_mode == "write") {
      require(cell_partition_count == 1u,
              "Reference generation requires G_cell = 1");
      require(tile_partition_count == 1u,
              "Reference generation requires G_tile = 1");
      require(visible_cuda_count == 1,
              "Reference generation requires exactly one visible GPU");
    }

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(current_case.natom);
    cuda_container<double> ry(current_case.natom);
    cuda_container<double> rz(current_case.natom);
    cuda_container<double> qc(current_case.natom);

    unsigned int grid_axis = 1u;
    while (static_cast<unsigned long long int>(grid_axis) *
               static_cast<unsigned long long int>(grid_axis) *
               static_cast<unsigned long long int>(grid_axis) <
           static_cast<unsigned long long int>(current_case.natom)) {
      require(grid_axis < std::numeric_limits<unsigned int>::max(),
              "Input grid axis exceeds unsigned int range");
      grid_axis++;
    }

    const unsigned long long int grid_plane =
        static_cast<unsigned long long int>(grid_axis) *
        static_cast<unsigned long long int>(grid_axis);
    const unsigned long long int grid_site_count =
        grid_plane * static_cast<unsigned long long int>(grid_axis);

    unsigned long long int permutation_multiplier = 104729ULL;
    permutation_multiplier %= grid_site_count;
    if (permutation_multiplier == 0ULL)
      permutation_multiplier = 1ULL;

    while (std::gcd(permutation_multiplier, grid_site_count) != 1ULL) {
      permutation_multiplier += 2ULL;
      permutation_multiplier %= grid_site_count;
      if (permutation_multiplier == 0ULL)
        permutation_multiplier = 1ULL;
    }

    const unsigned long long int permutation_offset = grid_site_count / 3ULL;
    const double spacing = current_case.box / static_cast<double>(grid_axis);

    for (unsigned int atom = 0; atom < current_case.natom; atom++) {
      const unsigned long long int site =
          (permutation_multiplier * static_cast<unsigned long long int>(atom) +
           permutation_offset) %
          grid_site_count;

      const unsigned int x = static_cast<unsigned int>(site / grid_plane);
      const unsigned long long int plane_offset = site % grid_plane;
      const unsigned int y =
          static_cast<unsigned int>(plane_offset / grid_axis);
      const unsigned int z =
          static_cast<unsigned int>(plane_offset % grid_axis);

      rx[atom] = (static_cast<double>(x) + 0.5) * spacing;
      ry[atom] = (static_cast<double>(y) + 0.5) * spacing;
      rz[atom] = (static_cast<double>(z) + 0.5) * spacing;
      qc[atom] = ((atom % 2u) == 0u) ? 1.0 : -1.0;
    }

    if ((current_case.natom % 2u) != 0u)
      qc[current_case.natom - 1u] = 0.0;

    rx.transfer_to_device();
    ry.transfer_to_device();
    rz.transfer_to_device();
    qc.transfer_to_device();

    glst_force force;
    force.set_gpu_layout(cell_partition_count, tile_partition_count);
    force.init(current_case.natom, current_case.tol, current_case.box,
               current_case.box, current_case.box, current_case.rcut);

    const glst_plan &plan = glst_force_test_access::plan(force);

    require(plan.natom() == current_case.natom,
            "Plan atom count does not match the test case");
    require(plan.ncell_x() == current_case.expected_ncell_axis,
            "Unexpected x-cell count");
    require(plan.ncell_y() == current_case.expected_ncell_axis,
            "Unexpected y-cell count");
    require(plan.ncell_z() == current_case.expected_ncell_axis,
            "Unexpected z-cell count");

    const unsigned long long int expected_ncell =
        static_cast<unsigned long long int>(current_case.expected_ncell_axis) *
        static_cast<unsigned long long int>(current_case.expected_ncell_axis) *
        static_cast<unsigned long long int>(current_case.expected_ncell_axis);

    require(static_cast<unsigned long long int>(plan.ncell()) == expected_ncell,
            "Unexpected total cell count");
    require(plan.ngroup() == current_case.expected_ngroup,
            "Unexpected alpha-group count");
    require(plan.cell_partition_count() == cell_partition_count,
            "Plan cell-partition count does not match G_cell");
    require(plan.tile_partition_count() == tile_partition_count,
            "Plan tile-partition count does not match G_tile");
    require(plan.tile_count() > 0u, "Plan contains no cubature tiles");
    require(plan.max_tile_nodes() > 0u, "Plan maximum tile-node count is zero");

    if (current_case.expected_total_nodes == 0u) {
      require(plan.tot_num_nodes() > 0u, "Plan contains no cubature nodes");
    } else {
      require(plan.tot_num_nodes() == current_case.expected_total_nodes,
              "Unexpected total cubature-node count");
    }

    require(!plan.num_nodes().empty(),
            "Plan contains no cubature node-count arrays");
    require(plan.num_nodes()[0].size() ==
                static_cast<std::size_t>(plan.ngroup()),
            "Cubature node-count array size does not match ngroup");

    std::vector<std::uint32_t> group_node_counts(plan.ngroup(), 0u);

    for (unsigned int group = 0; group < plan.ngroup(); group++) {
      const unsigned int node_count = plan.num_nodes()[0][group];
      group_node_counts[group] = static_cast<std::uint32_t>(node_count);

      if (!current_case.expected_group_nodes.empty()) {
        require(group < current_case.expected_group_nodes.size(),
                "Expected group-node array is too small");
        require(node_count == current_case.expected_group_nodes[group],
                "Unexpected cubature-node count in alpha group " +
                    std::to_string(group));
      }
    }

    if (!current_case.expected_group_nodes.empty()) {
      require(current_case.expected_group_nodes.size() ==
                  static_cast<std::size_t>(plan.ngroup()),
              "Expected group-node array size does not match ngroup");
    }

    std::cout << "RUN hybrid_layout_correctness: case=" << current_case.name
              << ", layout=" << cell_partition_count << "x"
              << tile_partition_count << ", visible_gpus=" << visible_cuda_count
              << ", reference_mode=" << reference_mode << std::endl;

    const ef_result first =
        run_calculation(force, rx, ry, rz, qc, current_case.natom);
    const ef_result second =
        run_calculation(force, rx, ry, rz, qc, current_case.natom);

    const std::size_t natom = static_cast<std::size_t>(current_case.natom);

    validate_component(first.fx, natom, "first fx");
    validate_component(first.fy, natom, "first fy");
    validate_component(first.fz, natom, "first fz");
    validate_component(first.en, natom, "first en");
    validate_component(second.fx, natom, "second fx");
    validate_component(second.fy, natom, "second fy");
    validate_component(second.fz, natom, "second fz");
    validate_component(second.en, natom, "second en");

    verify_repeat_component(first.fx, second.fx, "fx", current_case.name,
                            cell_partition_count, tile_partition_count);
    verify_repeat_component(first.fy, second.fy, "fy", current_case.name,
                            cell_partition_count, tile_partition_count);
    verify_repeat_component(first.fz, second.fz, "fz", current_case.name,
                            cell_partition_count, tile_partition_count);
    verify_repeat_component(first.en, second.en, "en", current_case.name,
                            cell_partition_count, tile_partition_count);

    ef_result loaded_reference;
    const ef_result *reference = nullptr;

    if (reference_mode == "write") {
      reference_header header;
      header.magic = reference_magic;
      header.version = reference_version;
      header.natom = static_cast<std::uint32_t>(plan.natom());
      header.ncell_x = static_cast<std::uint32_t>(plan.ncell_x());
      header.ncell_y = static_cast<std::uint32_t>(plan.ncell_y());
      header.ncell_z = static_cast<std::uint32_t>(plan.ncell_z());
      header.ngroup = static_cast<std::uint32_t>(plan.ngroup());
      header.total_nodes = static_cast<std::uint32_t>(plan.tot_num_nodes());
      header.max_tile_nodes = static_cast<std::uint32_t>(plan.max_tile_nodes());
      header.tile_count = static_cast<std::uint32_t>(plan.tile_count());
      header.tol = current_case.tol;
      header.box = current_case.box;
      header.rcut = current_case.rcut;

      std::ofstream output(reference_path, std::ios::binary | std::ios::trunc);
      require(static_cast<bool>(output),
              "Could not open private reference file for writing");

      write_scalar(output, header.magic, "reference magic");
      write_scalar(output, header.version, "reference version");
      write_scalar(output, header.natom, "reference atom count");
      write_scalar(output, header.ncell_x, "reference x-cell count");
      write_scalar(output, header.ncell_y, "reference y-cell count");
      write_scalar(output, header.ncell_z, "reference z-cell count");
      write_scalar(output, header.ngroup, "reference alpha-group count");
      write_scalar(output, header.total_nodes, "reference cubature-node count");
      write_scalar(output, header.max_tile_nodes,
                   "reference maximum tile-node count");
      write_scalar(output, header.tile_count, "reference cubature-tile count");
      write_scalar(output, header.tol, "reference tolerance");
      write_scalar(output, header.box, "reference box dimension");
      write_scalar(output, header.rcut, "reference cutoff");
      write_vector(output, group_node_counts, "reference group node counts");
      write_vector(output, first.fx, "reference fx");
      write_vector(output, first.fy, "reference fy");
      write_vector(output, first.fz, "reference fz");
      write_vector(output, first.en, "reference en");

      output.close();
      require(static_cast<bool>(output),
              "Could not finish writing private reference file");

      reference = &first;
    } else {
      std::ifstream input(reference_path, std::ios::binary);
      require(static_cast<bool>(input),
              "Could not open private reference file for reading");

      reference_header header = {};
      read_scalar(input, header.magic, "reference magic");
      read_scalar(input, header.version, "reference version");
      read_scalar(input, header.natom, "reference atom count");
      read_scalar(input, header.ncell_x, "reference x-cell count");
      read_scalar(input, header.ncell_y, "reference y-cell count");
      read_scalar(input, header.ncell_z, "reference z-cell count");
      read_scalar(input, header.ngroup, "reference alpha-group count");
      read_scalar(input, header.total_nodes, "reference cubature-node count");
      read_scalar(input, header.max_tile_nodes,
                  "reference maximum tile-node count");
      read_scalar(input, header.tile_count, "reference cubature-tile count");
      read_scalar(input, header.tol, "reference tolerance");
      read_scalar(input, header.box, "reference box dimension");
      read_scalar(input, header.rcut, "reference cutoff");

      require(header.magic == reference_magic,
              "Reference file has an invalid magic value");
      require(header.version == reference_version,
              "Reference file has an unsupported version");
      require(header.natom == static_cast<std::uint32_t>(plan.natom()),
              "Reference atom count does not match the current plan");
      require(header.ncell_x == static_cast<std::uint32_t>(plan.ncell_x()),
              "Reference x-cell count does not match the current plan");
      require(header.ncell_y == static_cast<std::uint32_t>(plan.ncell_y()),
              "Reference y-cell count does not match the current plan");
      require(header.ncell_z == static_cast<std::uint32_t>(plan.ncell_z()),
              "Reference z-cell count does not match the current plan");
      require(header.ngroup == static_cast<std::uint32_t>(plan.ngroup()),
              "Reference alpha-group count does not match the current plan");
      require(header.total_nodes ==
                  static_cast<std::uint32_t>(plan.tot_num_nodes()),
              "Reference cubature-node count does not match the current plan");
      require(header.max_tile_nodes ==
                  static_cast<std::uint32_t>(plan.max_tile_nodes()),
              "Reference maximum tile-node count does not match the current "
              "plan");
      require(header.tile_count ==
                  static_cast<std::uint32_t>(plan.tile_count()),
              "Reference cubature-tile count does not match the current plan");
      require(header.tol == current_case.tol,
              "Reference tolerance does not match the test case");
      require(header.box == current_case.box,
              "Reference box dimension does not match the test case");
      require(header.rcut == current_case.rcut,
              "Reference cutoff does not match the test case");

      std::vector<std::uint32_t> reference_group_node_counts(
          static_cast<std::size_t>(header.ngroup), 0u);
      read_vector(input, reference_group_node_counts,
                  "reference group node counts");
      require(reference_group_node_counts == group_node_counts,
              "Reference group node counts do not match the current plan");

      loaded_reference.fx.resize(natom);
      loaded_reference.fy.resize(natom);
      loaded_reference.fz.resize(natom);
      loaded_reference.en.resize(natom);

      read_vector(input, loaded_reference.fx, "reference fx");
      read_vector(input, loaded_reference.fy, "reference fy");
      read_vector(input, loaded_reference.fz, "reference fz");
      read_vector(input, loaded_reference.en, "reference en");

      const int trailing = input.peek();
      require(trailing == std::char_traits<char>::eof(),
              "Reference file contains trailing data");
      require(!input.bad(), "Reference file read failed");

      validate_component(loaded_reference.fx, natom, "reference fx");
      validate_component(loaded_reference.fy, natom, "reference fy");
      validate_component(loaded_reference.fz, natom, "reference fz");
      validate_component(loaded_reference.en, natom, "reference en");

      reference = &loaded_reference;
    }

    require(reference != nullptr, "Reference result is not available");

    const error_stats fx_stats =
        calculate_error_stats(first.fx, reference->fx, "fx");
    const error_stats fy_stats =
        calculate_error_stats(first.fy, reference->fy, "fy");
    const error_stats fz_stats =
        calculate_error_stats(first.fz, reference->fz, "fz");
    const error_stats en_stats =
        calculate_error_stats(first.en, reference->en, "en");

    std::cout << "Comparison tolerances: absolute=" << std::scientific
              << std::setprecision(6) << absolute_tolerance
              << ", relative=" << relative_tolerance << std::endl;
    print_error_stats("fx", fx_stats);
    print_error_stats("fy", fy_stats);
    print_error_stats("fz", fz_stats);
    print_error_stats("en", en_stats);

    enforce_component_tolerance("fx", fx_stats, current_case.name,
                                cell_partition_count, tile_partition_count);
    enforce_component_tolerance("fy", fy_stats, current_case.name,
                                cell_partition_count, tile_partition_count);
    enforce_component_tolerance("fz", fz_stats, current_case.name,
                                cell_partition_count, tile_partition_count);
    enforce_component_tolerance("en", en_stats, current_case.name,
                                cell_partition_count, tile_partition_count);

    const double overall_worst_ratio =
        std::max(std::max(fx_stats.worst_tolerance_ratio,
                          fy_stats.worst_tolerance_ratio),
                 std::max(fz_stats.worst_tolerance_ratio,
                          en_stats.worst_tolerance_ratio));

    std::cout << "PASS hybrid_layout_correctness layout: case="
              << current_case.name << ", layout=" << cell_partition_count << "x"
              << tile_partition_count << ", repeatability=bitwise-identical"
              << ", overall_worst_tolerance_ratio=" << std::scientific
              << std::setprecision(6) << overall_worst_ratio << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL hybrid_layout_correctness layout: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int run_orchestrator(void) {
  try {
    const std::string program = executable_path();
    temporary_file probe_file("probe");

    std::vector<std::pair<std::string, std::string>> probe_environment;
    probe_environment.push_back({stage_environment, "probe"});
    probe_environment.push_back({probe_path_environment, probe_file.path()});

    require_child_success(program, probe_environment,
                          "CUDA-device visibility probe");

    std::ifstream probe_input(probe_file.path());
    require(static_cast<bool>(probe_input),
            "Could not open CUDA-device probe result");

    int visible_cuda_count = 0;
    probe_input >> visible_cuda_count;
    require(static_cast<bool>(probe_input),
            "Could not parse CUDA-device probe result");
    require(visible_cuda_count > 0,
            "No CUDA-capable device is visible to the test");

    const std::vector<std::string> device_tokens =
        visible_device_tokens(visible_cuda_count);
    const std::string reference_device = join_device_tokens(device_tokens, 1u);
    const std::string all_devices = join_device_tokens(
        device_tokens, static_cast<std::size_t>(visible_cuda_count));
    const std::vector<gpu_layout> layouts =
        multiple_gpu_layouts(static_cast<unsigned int>(visible_cuda_count));
    const std::vector<test_case> cases = test_cases();

    bool has_true_mode3_layout = false;
    for (std::size_t i = 0; i < layouts.size(); i++) {
      if ((layouts[i].cell_partition_count > 1u) &&
          (layouts[i].tile_partition_count > 1u)) {
        has_true_mode3_layout = true;
        break;
      }
    }

    std::cout << "RUN hybrid_layout_correctness: visible_gpus="
              << visible_cuda_count << ", cases=" << cases.size()
              << ", multi_gpu_layouts=" << layouts.size() << std::endl;

    if ((visible_cuda_count > 1) && !has_true_mode3_layout) {
      std::cout << "NOTE hybrid_layout_correctness: " << visible_cuda_count
                << " visible GPUs have no nontrivial G_cell x G_tile "
                   "factorization; pure-cell and pure-tile layouts will be "
                   "tested"
                << std::endl;
    }

    std::size_t completed_layout_runs = 0;

    for (std::size_t case_index = 0; case_index < cases.size(); case_index++) {
      const test_case &current_case = cases[case_index];
      temporary_file reference_file("reference_" + current_case.selector);

      std::vector<std::pair<std::string, std::string>> reference_environment;
      reference_environment.push_back({stage_environment, "layout"});
      reference_environment.push_back(
          {case_environment, current_case.selector});
      reference_environment.push_back({cell_partition_environment, "1"});
      reference_environment.push_back({tile_partition_environment, "1"});
      reference_environment.push_back({reference_mode_environment, "write"});
      reference_environment.push_back(
          {reference_path_environment, reference_file.path()});
      reference_environment.push_back(
          {"CUDA_VISIBLE_DEVICES", reference_device});

      require_child_success(program, reference_environment,
                            current_case.name + " 1x1 reference");
      completed_layout_runs++;

      for (std::size_t layout_index = 0; layout_index < layouts.size();
           layout_index++) {
        const gpu_layout &layout = layouts[layout_index];

        std::vector<std::pair<std::string, std::string>> layout_environment;
        layout_environment.push_back({stage_environment, "layout"});
        layout_environment.push_back({case_environment, current_case.selector});
        layout_environment.push_back(
            {cell_partition_environment,
             std::to_string(layout.cell_partition_count)});
        layout_environment.push_back(
            {tile_partition_environment,
             std::to_string(layout.tile_partition_count)});
        layout_environment.push_back({reference_mode_environment, "compare"});
        layout_environment.push_back(
            {reference_path_environment, reference_file.path()});
        layout_environment.push_back({"CUDA_VISIBLE_DEVICES", all_devices});

        std::ostringstream description;
        description << current_case.name << " " << layout.cell_partition_count
                    << "x" << layout.tile_partition_count << " layout";

        require_child_success(program, layout_environment, description.str());
        completed_layout_runs++;
      }

      std::cout << "PASS hybrid_layout_correctness case: " << current_case.name
                << std::endl;
    }

    std::cout << "PASS hybrid_layout_correctness: visible_gpus="
              << visible_cuda_count
              << ", completed_layout_runs=" << completed_layout_runs
              << ", cases=" << cases.size() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL hybrid_layout_correctness: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(void) {
  const char *stage = std::getenv(stage_environment);

  if ((stage == nullptr) || (stage[0] == '\0'))
    return run_orchestrator();

  if (std::strcmp(stage, "probe") == 0)
    return run_probe_child();

  if (std::strcmp(stage, "layout") == 0)
    return run_layout_child();

  std::cerr << "FAIL hybrid_layout_correctness: unknown internal stage "
            << stage << std::endl;
  return EXIT_FAILURE;
}
