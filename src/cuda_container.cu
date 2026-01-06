// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "cuda_container.hcu"

#include "cuda_utils.hcu"

template <typename T>
cuda_container<T>::cuda_container(void) : h_array_(), d_array_() {}

template <typename T>
cuda_container<T>::cuda_container(const std::size_t count)
    : h_array_(count), d_array_(count) {}

template <typename T>
cuda_container<T>::cuda_container(const std::vector<T> &other)
    : h_array_(other), d_array_(other.size()) {
  this->transfer_to_device();
}

template <typename T>
cuda_container<T>::cuda_container(const std::vector<T> &&other)
    : h_array_(other), d_array_(other.size()) {
  this->transfer_to_device();
}

template <typename T>
cuda_container<T>::cuda_container(const device_vector<T> &other)
    : h_array_(other.size()), d_array_(other) {
  this->transfer_to_host();
}

template <typename T>
cuda_container<T>::cuda_container(const device_vector<T> &&other)
    : h_array_(other.size()), d_array_(other) {
  this->transfer_to_host();
}

template <typename T>
cuda_container<T>::cuda_container(const cuda_container<T> &other)
    : h_array_(other.h_array()), d_array_(other.d_array()) {}

template <typename T>
cuda_container<T>::cuda_container(const cuda_container<T> &&other)
    : h_array_(other.h_array()), d_array_(other.d_array()) {}

template <typename T>
cuda_container<T> &cuda_container<T>::operator=(const std::vector<T> &other) {
  this->h_array_ = other;
  this->d_array_.resize(other.size());
  this->transfer_to_device();
  return *this;
}

template <typename T>
cuda_container<T> &cuda_container<T>::operator=(const std::vector<T> &&other) {
  this->h_array_ = other;
  this->d_array_.resize(other.size());
  this->transfer_to_device();
  return *this;
}

template <typename T>
cuda_container<T> &cuda_container<T>::operator=(const device_vector<T> &other) {
  this->h_array_.resize(other.size());
  this->d_array_ = other;
  this->transfer_to_host();
  return *this;
}

template <typename T>
cuda_container<T> &
cuda_container<T>::operator=(const device_vector<T> &&other) {
  this->h_array_.resize(other.size());
  this->d_array_ = other;
  this->transfer_to_host();
  return *this;
}

template <typename T>
cuda_container<T> &
cuda_container<T>::operator=(const cuda_container<T> &other) {
  this->h_array_ = other.h_array();
  this->d_array_ = other.d_array();
  return *this;
}

template <typename T>
cuda_container<T> &
cuda_container<T>::operator=(const cuda_container<T> &&other) {
  this->h_array_ = other.h_array();
  this->d_array_ = other.d_array();
  return *this;
}

template <typename T>
const T &cuda_container<T>::at(const std::size_t pos) const {
  return this->h_array_.at(pos);
}

template <typename T> T &cuda_container<T>::at(const std::size_t pos) {
  return this->h_array_.at(pos);
}

template <typename T>
const T &cuda_container<T>::operator[](const std::size_t pos) const {
  return this->h_array_[pos];
}

template <typename T> T &cuda_container<T>::operator[](const std::size_t pos) {
  return this->h_array_[pos];
}

template <typename T>
const std::vector<T> &cuda_container<T>::h_array(void) const {
  return this->h_array_;
}

template <typename T> std::vector<T> &cuda_container<T>::h_array(void) {
  return this->h_array_;
}

template <typename T>
const device_vector<T> &cuda_container<T>::d_array(void) const {
  return this->d_array_;
}

template <typename T> device_vector<T> &cuda_container<T>::d_array(void) {
  return this->d_array_;
}

template <typename T> std::size_t cuda_container<T>::size(void) const {
  return this->h_array_.size();
}

template <typename T> void cuda_container<T>::clear(void) {
  this->h_array_.clear();
  this->d_array_.clear();
  return;
}

template <typename T> void cuda_container<T>::resize(const std::size_t count) {
  this->h_array_.resize(count);
  this->d_array_.resize(count);
  return;
}

template <typename T>
void cuda_container<T>::set(const std::vector<T> &values) {
  this->h_array_ = values;
  this->d_array_.resize(values.size());
  this->transfer_to_device();
  return;
}

template <typename T>
void cuda_container<T>::set(const device_vector<T> &values) {
  this->h_array_.resize(values.size());
  this->d_array_ = values;
  this->transfer_to_host();
  return;
}

template <typename T> void cuda_container<T>::set(const T &value) {
  this->h_array_.assign(this->h_array_.size(), value);
  this->transfer_to_device();
  return;
}

template <typename T> void cuda_container<T>::transfer_to_device(void) {
  cudaCheck(cudaMemcpy(static_cast<void *>(this->d_array_.data()),
                       static_cast<const void *>(this->h_array_.data()),
                       this->h_array_.size() * sizeof(T),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaDeviceSynchronize());
  return;
}

template <typename T> void cuda_container<T>::transfer_to_host(void) {
  cudaCheck(cudaMemcpy(static_cast<void *>(this->h_array_.data()),
                       static_cast<const void *>(this->d_array_.data()),
                       this->d_array_.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());
  return;
}
