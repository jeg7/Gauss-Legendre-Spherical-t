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

#include <type_traits>

template <typename CT> glst_plan<CT>::glst_plan(void) {
  static_assert(std::is_floating_point_v<CT>,
                "CT must be a floating-point type (float or double)");
}

template <typename CT> glst_plan<CT>::~glst_plan(void) {}

template class glst_plan<float>;
template class glst_plan<double>;
