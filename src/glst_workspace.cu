// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "glst_workspace.hcu"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

inline static std::size_t checked_mul(const std::size_t a, const std::size_t b,
                                      const std::string_view label) {
  if ((a != 0) && (b > std::numeric_limits<std::size_t>::max() / a)) {
    const std::string s(label);
    throw std::runtime_error(
        "FATAL ERROR: glst_workspace: Overflow while computing " + s);
  }
  return (a * b);
}

glst_workspace::glst_workspace(void)
    : atom_capacity_(0), cell_capacity_(0), tile_node_capacity_(0),
      tile_buffer_capacity_(0), idx_(), sorted_idx_(), rx_(), ry_(), rz_(),
      qc_(), packets_(), sorted_packets_(), atom_cell_idx_(),
      atom_cell_sorted_idx_(), fx_(), fy_(), fz_(), en_(), cell_atom_point_(),
      cell_atom_count_(), max_atoms_cell_(), sf_re_(), sf_im_(), rmt_sum_re_(),
      rmt_sum_im_() {}

glst_workspace::glst_workspace(const glst_plan &plan) : glst_workspace() {
  this->init(plan);
}

glst_workspace::~glst_workspace(void) {}

std::size_t glst_workspace::atom_capacity(void) const {
  return this->atom_capacity_;
}

std::size_t glst_workspace::cell_capacity(void) const {
  return this->cell_capacity_;
}

std::size_t glst_workspace::tile_node_capacity(void) const {
  return this->tile_node_capacity_;
}

std::size_t glst_workspace::tile_buffer_capacity(void) const {
  return this->tile_buffer_capacity_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::idx(void) const {
  return this->idx_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::sorted_idx(void) const {
  return this->sorted_idx_;
}

const std::vector<cuda_container<double>> &glst_workspace::rx(void) const {
  return this->rx_;
}

const std::vector<cuda_container<double>> &glst_workspace::ry(void) const {
  return this->ry_;
}

const std::vector<cuda_container<double>> &glst_workspace::rz(void) const {
  return this->rz_;
}

const std::vector<cuda_container<double>> &glst_workspace::qc(void) const {
  return this->qc_;
}

const std::vector<cuda_container<atom_packet>> &
glst_workspace::packets(void) const {
  return this->packets_;
}

const std::vector<cuda_container<atom_packet>> &
glst_workspace::sorted_packets(void) const {
  return this->sorted_packets_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_idx(void) const {
  return this->atom_cell_idx_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_sorted_idx(void) const {
  return this->atom_cell_sorted_idx_;
}

const std::vector<cuda_container<double>> &glst_workspace::fx(void) const {
  return this->fx_;
}

const std::vector<cuda_container<double>> &glst_workspace::fy(void) const {
  return this->fy_;
}

const std::vector<cuda_container<double>> &glst_workspace::fz(void) const {
  return this->fz_;
}

const std::vector<cuda_container<double>> &glst_workspace::en(void) const {
  return this->en_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_point(void) const {
  return this->cell_atom_point_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_count(void) const {
  return this->cell_atom_count_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::max_atoms_cell(void) const {
  return this->max_atoms_cell_;
}

const std::vector<cuda_container<double>> &glst_workspace::sf_re(void) const {
  return this->sf_re_;
}

const std::vector<cuda_container<double>> &glst_workspace::sf_im(void) const {
  return this->sf_im_;
}

const std::vector<cuda_container<double>> &
glst_workspace::rmt_sum_re(void) const {
  return this->rmt_sum_re_;
}

const std::vector<cuda_container<double>> &
glst_workspace::rmt_sum_im(void) const {
  return this->rmt_sum_im_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::idx(void) {
  return this->idx_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::sorted_idx(void) {
  return this->sorted_idx_;
}

std::vector<cuda_container<double>> &glst_workspace::rx(void) {
  return this->rx_;
}

std::vector<cuda_container<double>> &glst_workspace::ry(void) {
  return this->ry_;
}

std::vector<cuda_container<double>> &glst_workspace::rz(void) {
  return this->rz_;
}

std::vector<cuda_container<double>> &glst_workspace::qc(void) {
  return this->qc_;
}

std::vector<cuda_container<atom_packet>> &glst_workspace::packets(void) {
  return this->packets_;
}

std::vector<cuda_container<atom_packet>> &glst_workspace::sorted_packets(void) {
  return this->sorted_packets_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::atom_cell_idx(void) {
  return this->atom_cell_idx_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_sorted_idx(void) {
  return this->atom_cell_sorted_idx_;
}

std::vector<cuda_container<double>> &glst_workspace::fx(void) {
  return this->fx_;
}

std::vector<cuda_container<double>> &glst_workspace::fy(void) {
  return this->fy_;
}

std::vector<cuda_container<double>> &glst_workspace::fz(void) {
  return this->fz_;
}

std::vector<cuda_container<double>> &glst_workspace::en(void) {
  return this->en_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_point(void) {
  return this->cell_atom_point_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_count(void) {
  return this->cell_atom_count_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::max_atoms_cell(void) {
  return this->max_atoms_cell_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_re(void) {
  return this->sf_re_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_im(void) {
  return this->sf_im_;
}

std::vector<cuda_container<double>> &glst_workspace::rmt_sum_re(void) {
  return this->rmt_sum_re_;
}

std::vector<cuda_container<double>> &glst_workspace::rmt_sum_im(void) {
  return this->rmt_sum_im_;
}

void glst_workspace::init(const glst_plan &plan) {
  this->clear();

  const std::size_t natom = static_cast<std::size_t>(plan.natom());
  const std::size_t ncell = static_cast<std::size_t>(plan.ncell());
  const std::size_t max_tile_nodes =
      static_cast<std::size_t>(plan.max_tile_nodes());

  if (natom == 0)
    throw std::runtime_error("FATAL ERROR: glst_workspace::init: natom is 0");

  if (ncell == 0)
    throw std::runtime_error("FATAL ERROR: glst_workspace::init: ncell is 0");

  if (max_tile_nodes == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_workspace::init: max_tile_nodes is 0");
  }

  const std::size_t tile_buffer_nodes =
      checked_mul(ncell, max_tile_nodes, "tile buffer node count");

  this->atom_capacity_ = natom;
  this->cell_capacity_ = ncell;
  this->tile_node_capacity_ = max_tile_nodes;
  this->tile_buffer_capacity_ = tile_buffer_nodes;

  // JEG260714: Use vector-of-one storage for the initial single-GPU tiled path
  this->idx_.resize(1);
  this->sorted_idx_.resize(1);
  this->rx_.resize(1);
  this->ry_.resize(1);
  this->rz_.resize(1);
  this->qc_.resize(1);
  this->packets_.resize(1);
  this->sorted_packets_.resize(1);
  this->atom_cell_idx_.resize(1);
  this->atom_cell_sorted_idx_.resize(1);

  this->fx_.resize(1);
  this->fy_.resize(1);
  this->fz_.resize(1);
  this->en_.resize(1);

  this->cell_atom_point_.resize(1);
  this->cell_atom_count_.resize(1);
  this->max_atoms_cell_.resize(1);

  this->sf_re_.resize(1);
  this->sf_im_.resize(1);
  this->rmt_sum_re_.resize(1);
  this->rmt_sum_im_.resize(1);

  this->idx_[0].resize(this->atom_capacity_);
  this->sorted_idx_[0].resize(this->atom_capacity_);
  this->rx_[0].resize(this->atom_capacity_);
  this->ry_[0].resize(this->atom_capacity_);
  this->rz_[0].resize(this->atom_capacity_);
  this->qc_[0].resize(this->atom_capacity_);
  this->packets_[0].resize(this->atom_capacity_);
  this->sorted_packets_[0].resize(this->atom_capacity_);
  this->atom_cell_idx_[0].resize(this->atom_capacity_);
  this->atom_cell_sorted_idx_[0].resize(this->atom_capacity_);

  this->fx_[0].resize(this->atom_capacity_);
  this->fy_[0].resize(this->atom_capacity_);
  this->fz_[0].resize(this->atom_capacity_);
  this->en_[0].resize(this->atom_capacity_);

  this->cell_atom_point_[0].resize(this->cell_capacity_);
  this->cell_atom_count_[0].resize(this->cell_capacity_);
  this->max_atoms_cell_[0].resize(1);

  this->sf_re_[0].resize(this->tile_buffer_capacity_);
  this->sf_im_[0].resize(this->tile_buffer_capacity_);
  this->rmt_sum_re_[0].resize(this->tile_buffer_capacity_);
  this->rmt_sum_im_[0].resize(this->tile_buffer_capacity_);

  return;
}

void glst_workspace::clear(void) {
  this->atom_capacity_ = 0;
  this->cell_capacity_ = 0;
  this->tile_node_capacity_ = 0;
  this->tile_buffer_capacity_ = 0;

  this->idx_.clear();
  this->sorted_idx_.clear();
  this->rx_.clear();
  this->ry_.clear();
  this->rz_.clear();
  this->qc_.clear();
  this->packets_.clear();
  this->sorted_packets_.clear();
  this->atom_cell_idx_.clear();
  this->atom_cell_sorted_idx_.clear();

  this->fx_.clear();
  this->fy_.clear();
  this->fz_.clear();
  this->en_.clear();

  this->cell_atom_point_.clear();
  this->cell_atom_count_.clear();
  this->max_atoms_cell_.clear();

  this->sf_re_.clear();
  this->sf_im_.clear();
  this->rmt_sum_re_.clear();
  this->rmt_sum_im_.clear();

  return;
}
