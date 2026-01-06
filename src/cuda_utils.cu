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

void enable_p2p(const int cuda_count) {
  for (int device = 0; device < cuda_count; device++) {
    cudaCheck(cudaSetDevice(device));
    for (int peer_device = 0; peer_device < cuda_count; peer_device++) {
      if (device == peer_device) // Don't need to enable comm. with self
        continue;

      // Check if device can access memory on peer_device
      int can_access_peer = -1;
      cudaCheck(cudaDeviceCanAccessPeer(&can_access_peer, device, peer_device));
      if (can_access_peer != 1) {
        throw std::runtime_error(
            "Peer-peer access is not available between devices " +
            std::to_string(device) + " and " + std::to_string(peer_device));
      }

      // Enable peer-peer access
      cudaCheck(cudaDeviceEnablePeerAccess(peer_device, 0));
    }
  }

  return;
}

void disable_p2p(const int cuda_count) {
  for (int device = 0; device < cuda_count; device++) {
    cudaCheck(cudaSetDevice(device));
    for (int peer_device = 0; peer_device < cuda_count; peer_device++) {
      if (device == peer_device) // Don't need to disable comm. with self
        continue;

      // Disable peer-peer access
      cudaCheck(cudaDeviceDisablePeerAccess(peer_device));
    }
  }

  return;
}
