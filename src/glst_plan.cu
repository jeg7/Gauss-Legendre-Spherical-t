// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENS

#include "glst_plan.hcu"

glst_plan::glst_plan(void)
    : natom_(0), box_dim_x_(0.0), box_dim_y_(0.0), box_dim_z_(0.0), ncell_x_(0),
      ncell_y_(0), ncell_z_(0), ncell_(0), cell_dim_x_(0.0), cell_dim_y_(0.0),
      cell_dim_z_(0.0) {}

glst_plan::~glst_plan(void) {}

unsigned int glst_plan::natom(void) const { return this->natom_; }

double glst_plan::box_dim_x(void) const { return this->box_dim_x_; }

double glst_plan::box_dim_y(void) const { return this->box_dim_y_; }

double glst_plan::box_dim_z(void) const { return this->box_dim_z_; }

unsigned int glst_plan::ncell_x(void) const { return this->ncell_x_; }

unsigned int glst_plan::ncell_y(void) const { return this->ncell_y_; }

unsigned int glst_plan::ncell_z(void) const { return this->ncell_z_; }

unsigned int glst_plan::ncell(void) const { return this->ncell_; }

double glst_plan::cell_dim_x(void) const { return this->cell_dim_x_; }

double glst_plan::cell_dim_y(void) const { return this->cell_dim_y_; }

double glst_plan::cell_dim_z(void) const { return this->cell_dim_z_; }

void glst_plan::init_cells(const unsigned int natom, const double box_dim_x,
                           const double box_dim_y, const double box_dim_z,
                           const double rcut) {
  this->natom_ = natom;

  this->box_dim_x_ = box_dim_x;
  this->box_dim_y_ = box_dim_y;
  this->box_dim_z_ = box_dim_z;

  this->cell_dim_x_ = rcut;
  this->cell_dim_y_ = rcut;
  this->cell_dim_z_ = rcut;

  this->ncell_x_ = static_cast<unsigned int>(box_dim_x / this->cell_dim_x_);
  this->ncell_y_ = static_cast<unsigned int>(box_dim_y / this->cell_dim_y_);
  this->ncell_z_ = static_cast<unsigned int>(box_dim_z / this->cell_dim_z_);

  if (static_cast<double>(this->ncell_x_) * this->cell_dim_x_ < box_dim_x)
    this->ncell_x_++;
  if (static_cast<double>(this->ncell_y_) * this->cell_dim_y_ < box_dim_y)
    this->ncell_y_++;
  if (static_cast<double>(this->ncell_z_) * this->cell_dim_z_ < box_dim_z)
    this->ncell_z_++;

  this->ncell_ = this->ncell_x_ * this->ncell_y_ * this->ncell_z_;

  return;
}
