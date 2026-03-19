// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "cuda_utils.hcu"
#include "device_comm.hcu"

void nccl_all_reduce_sum_ip(std::vector<cuda_container<float>> &buffs,
                            std::size_t count, std::vector<ncclComm_t> &comms,
                            std::vector<cudaStream_t> &streams,
                            const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs[dev].d_array().data()),
                      static_cast<void *>(buffs[dev].d_array().data()), count,
                      ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_all_reduce_sum_ip(std::vector<cuda_container<double>> &buffs,
                            std::size_t count, std::vector<ncclComm_t> &comms,
                            std::vector<cudaStream_t> &streams,
                            const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs[dev].d_array().data()),
                      static_cast<void *>(buffs[dev].d_array().data()), count,
                      ncclDouble, ncclSum, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_all_reduce_sum_pair_ip(std::vector<cuda_container<float>> &buffs0,
                                 std::vector<cuda_container<float>> &buffs1,
                                 std::size_t count,
                                 std::vector<ncclComm_t> &comms,
                                 std::vector<cudaStream_t> &streams,
                                 const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs0[dev].d_array().data()),
                      static_cast<void *>(buffs0[dev].d_array().data()), count,
                      ncclFloat, ncclSum, comms[dev], streams[dev]));
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs1[dev].d_array().data()),
                      static_cast<void *>(buffs1[dev].d_array().data()), count,
                      ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_all_reduce_sum_pair_ip(std::vector<cuda_container<double>> &buffs0,
                                 std::vector<cuda_container<double>> &buffs1,
                                 std::size_t count,
                                 std::vector<ncclComm_t> &comms,
                                 std::vector<cudaStream_t> &streams,
                                 const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs0[dev].d_array().data()),
                      static_cast<void *>(buffs0[dev].d_array().data()), count,
                      ncclDouble, ncclSum, comms[dev], streams[dev]));
    ncclCheck(
        ncclAllReduce(static_cast<const void *>(buffs1[dev].d_array().data()),
                      static_cast<void *>(buffs1[dev].d_array().data()), count,
                      ncclDouble, ncclSum, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_all_reduce_sum_ef_ip(std::vector<cuda_container<float>> &fx,
                               std::vector<cuda_container<float>> &fy,
                               std::vector<cuda_container<float>> &fz,
                               std::vector<cuda_container<float>> &en,
                               std::size_t count,
                               std::vector<ncclComm_t> &comms,
                               std::vector<cudaStream_t> &streams,
                               const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(ncclAllReduce(static_cast<const void *>(fx[dev].d_array().data()),
                            static_cast<void *>(fx[dev].d_array().data()),
                            count, ncclFloat, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(fy[dev].d_array().data()),
                            static_cast<void *>(fy[dev].d_array().data()),
                            count, ncclFloat, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(fz[dev].d_array().data()),
                            static_cast<void *>(fz[dev].d_array().data()),
                            count, ncclFloat, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(en[dev].d_array().data()),
                            static_cast<void *>(en[dev].d_array().data()),
                            count, ncclFloat, ncclSum, comms[dev],
                            streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_all_reduce_sum_ef_ip(std::vector<cuda_container<double>> &fx,
                               std::vector<cuda_container<double>> &fy,
                               std::vector<cuda_container<double>> &fz,
                               std::vector<cuda_container<double>> &en,
                               std::size_t count,
                               std::vector<ncclComm_t> &comms,
                               std::vector<cudaStream_t> &streams,
                               const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(ncclAllReduce(static_cast<const void *>(fx[dev].d_array().data()),
                            static_cast<void *>(fx[dev].d_array().data()),
                            count, ncclDouble, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(fy[dev].d_array().data()),
                            static_cast<void *>(fy[dev].d_array().data()),
                            count, ncclDouble, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(fz[dev].d_array().data()),
                            static_cast<void *>(fz[dev].d_array().data()),
                            count, ncclDouble, ncclSum, comms[dev],
                            streams[dev]));
    ncclCheck(ncclAllReduce(static_cast<const void *>(en[dev].d_array().data()),
                            static_cast<void *>(en[dev].d_array().data()),
                            count, ncclDouble, ncclSum, comms[dev],
                            streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_root_reduce_sum_ef_ip(std::vector<cuda_container<float>> &fx,
                                std::vector<cuda_container<float>> &fy,
                                std::vector<cuda_container<float>> &fz,
                                std::vector<cuda_container<float>> &en,
                                std::size_t count,
                                std::vector<ncclComm_t> &comms,
                                std::vector<cudaStream_t> &streams,
                                const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(ncclReduce(static_cast<const void *>(fx[dev].d_array().data()),
                         static_cast<void *>(fx[0].d_array().data()), count,
                         ncclFloat, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(fy[dev].d_array().data()),
                         static_cast<void *>(fy[0].d_array().data()), count,
                         ncclFloat, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(fz[dev].d_array().data()),
                         static_cast<void *>(fz[0].d_array().data()), count,
                         ncclFloat, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(en[dev].d_array().data()),
                         static_cast<void *>(en[0].d_array().data()), count,
                         ncclFloat, ncclSum, 0, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}

void nccl_root_reduce_sum_ef_ip(std::vector<cuda_container<double>> &fx,
                                std::vector<cuda_container<double>> &fy,
                                std::vector<cuda_container<double>> &fz,
                                std::vector<cuda_container<double>> &en,
                                std::size_t count,
                                std::vector<ncclComm_t> &comms,
                                std::vector<cudaStream_t> &streams,
                                const int cuda_count) {
  ncclCheck(ncclGroupStart());
  for (int dev = 0; dev < cuda_count; dev++) {
    ncclCheck(ncclReduce(static_cast<const void *>(fx[dev].d_array().data()),
                         static_cast<void *>(fx[0].d_array().data()), count,
                         ncclDouble, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(fy[dev].d_array().data()),
                         static_cast<void *>(fy[0].d_array().data()), count,
                         ncclDouble, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(fz[dev].d_array().data()),
                         static_cast<void *>(fz[0].d_array().data()), count,
                         ncclDouble, ncclSum, 0, comms[dev], streams[dev]));
    ncclCheck(ncclReduce(static_cast<const void *>(en[dev].d_array().data()),
                         static_cast<void *>(en[0].d_array().data()), count,
                         ncclDouble, ncclSum, 0, comms[dev], streams[dev]));
  }
  ncclCheck(ncclGroupEnd());
  return;
}
