// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "device_vector.hcu"

#include "cuda_utils.hcu"

template <typename T>
device_vector<T>::device_vector(void)
    : size_(0), capacity_(0), data_(nullptr) {}

template <typename T>
device_vector<T>::device_vector(const std::size_t count) : device_vector() {
  this->allocate(count);
  this->size_ = count;
}

template <typename T>
device_vector<T>::device_vector(const device_vector<T> &other)
    : device_vector() {
  this->allocate(other.capacity());
  this->size_ = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(this->data_),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
device_vector<T>::device_vector(const device_vector<T> &&other)
    : device_vector() {
  this->allocate(other.capacity());
  this->size_ = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(this->data_),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T> device_vector<T>::~device_vector(void) {
  this->deallocate();
}

template <typename T>
device_vector<T> &device_vector<T>::operator=(const device_vector<T> &other) {
  this->reallocate(other.capacity());
  this->size_ = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(this->data_),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T>
device_vector<T> &device_vector<T>::operator=(const device_vector<T> &&other) {
  this->reallocate(other.capacity());
  this->size_ = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(this->data_),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  return *this;
}

template <typename T> const T *device_vector<T>::data(void) const {
  return this->data_;
}

template <typename T> T *device_vector<T>::data(void) { return this->data_; }

template <typename T> bool device_vector<T>::empty(void) const {
  return (this->size_ == 0);
}

template <typename T> std::size_t device_vector<T>::size(void) const {
  return this->size_;
}

template <typename T> std::size_t device_vector<T>::capacity(void) const {
  return this->capacity_;
}

template <typename T> void device_vector<T>::shrink_to_fit(void) {
  this->reallocate(this->size_);
  return;
}

template <typename T> void device_vector<T>::clear(void) {
  this->deallocate();
  return;
}

template <typename T>
__global__ static void set_back_kernel(T *data, const std::size_t size,
                                       const T value) {
  if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
      (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
    data[size - 1] = value;
  return;
}

template <typename T> void device_vector<T>::push_back(const T &value) {
  if (this->size_ >= this->capacity_) // Increase size of memory block by 50%
    this->reallocate(this->capacity_ + this->capacity_ / 2);

  this->size_++;

  set_back_kernel<<<1, 1>>>(this->data_, this->size_, value);

  return;
}

template <typename T> void device_vector<T>::resize(const std::size_t count) {
  if (this->capacity_ == 0)
    this->allocate(count);
  else if (this->capacity_ < count)
    this->reallocate(count);
  this->size_ = count;
  return;
}

template <typename T> void device_vector<T>::swap(device_vector<T> &other) {
  // Copy properties and data of this device_vector
  std::size_t size = this->size_;
  std::size_t capacity = this->capacity_;
  T *data = nullptr;
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data), capacity * sizeof(T)));
  cudaCheck(cudaMemcpy(static_cast<void *>(data),
                       static_cast<const void *>(this->data_),
                       this->capacity_ * sizeof(T), cudaMemcpyDeviceToDevice));

  // Copy properties and data from other device_vector to this device_vector
  this->reallocate(other.capacity());
  this->size_ = other.size();
  cudaCheck(cudaMemcpy(static_cast<void *>(this->data_),
                       static_cast<const void *>(other.data()),
                       other.size() * sizeof(T), cudaMemcpyDeviceToDevice));

  // Copy properties and data from this device_vector to other device_vector
  other.reallocate(capacity);
  other.resize(size);
  cudaCheck(cudaMemcpy(static_cast<void *>(other.data()),
                       static_cast<const void *>(data), size * sizeof(T),
                       cudaMemcpyDeviceToDevice));

  // Free temporary memory used for copying data
  cudaCheck(cudaFree(static_cast<void *>(data)));

  return;
}

template <typename T> void device_vector<T>::allocate(const std::size_t count) {
  cudaCheck(
      cudaMalloc(reinterpret_cast<void **>(&(this->data_)), count * sizeof(T)));
  this->capacity_ = count;
  return;
}

template <typename T>
void device_vector<T>::reallocate(const std::size_t count) {
  if (count == this->capacity_) // No need for a new memory block
    return;

  // Allocate new memory block
  std::size_t oldSize = this->size_;
  T *data = nullptr;
  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));

  // Copy relevant data to new memory block
  cudaCheck(cudaMemcpy(
      static_cast<void *>(data), static_cast<const void *>(this->data_),
      ((count < this->size_) ? count : this->size_) * sizeof(T),
      cudaMemcpyDeviceToDevice));

  // Free old memory block
  this->deallocate();

  // Assign new memory block
  this->size_ = (count < oldSize) ? count : oldSize;
  this->capacity_ = count;
  this->data_ = data;

  return;
}

template <typename T> void device_vector<T>::deallocate(void) {
  this->size_ = 0;
  this->capacity_ = 0;
  if (this->data_ != nullptr) {
    cudaCheck(cudaFree(static_cast<void *>(this->data_)));
    this->data_ = nullptr;
  }
  return;
}
