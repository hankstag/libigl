// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_COMISO_NROSY_H
#define IGL_COMISO_NROSY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../../igl_inline.h"

namespace igl
{
  namespace copyleft
  {
  namespace comiso
  {
    // Generate a N-RoSy field from a sparse set of constraints
    //
    // Inputs:
    //   V       #V by 3 list of mesh vertex coordinates
    //   F       #F by 3 list of mesh faces (must be triangles)
    //   b       #B by 1 list of constrained face indices
    //   bc      #B by 3 list of representative vectors for the constrained
    //     faces
    //   b_soft  #S by 1 b for soft constraints
    //   w_soft  #S by 1 weight for the soft constraints (0-1)
    //   bc_soft #S by 3 bc for soft constraints
    //   N       the degree of the N-RoSy vector field
    //   soft    the strength of the soft constraints w.r.t. smoothness
    //           (0 -> smoothness only, 1->constraints only)
    // Outputs:
    //   R       #F by 3 the representative vectors of the interpolated field
    //   S       #V by 1 the singularity index for each vertex (0 = regular)
    IGL_INLINE void nrosy(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::VectorXi& b,
      const Eigen::MatrixXd& bc,
      const Eigen::VectorXi& b_soft,
      const Eigen::VectorXd& w_soft,
      const Eigen::MatrixXd& bc_soft,
      int N,
      double soft,
      Eigen::MatrixXd& R,
      Eigen::VectorXd& S
      );
    //wrapper for the case without soft constraints
    IGL_INLINE void nrosy(
     const Eigen::MatrixXd& V,
     const Eigen::MatrixXi& F,
     const Eigen::VectorXi& b,
     const Eigen::MatrixXd& bc,
     int N,
     Eigen::MatrixXd& R,
     Eigen::VectorXd& S
      );
    
    // extension of nrosy to support following
    // - customized reference frames (rather than first edge by default)
    // - linear constraints for period jumps
    // - access to internal variables (pj, kappa)
    IGL_INLINE void nrosy(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::VectorXi& b,
      const Eigen::MatrixXd& bc,
      const std::vector<Eigen::MatrixXd>& TP_in, // optional - leave empty if want to use default
      int N,
      Eigen::MatrixXd& R,
      Eigen::VectorXd& S,
      Eigen::VectorXi& pj,
      Eigen::VectorXd& kappa
    );
    
    // new interface where the field defined
    // is represented as angles rather than vectors
      IGL_INLINE void nrosy(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::VectorXi& b,
      const Eigen::VectorXd& bc,
      const std::vector<Eigen::MatrixXd>& TP_set,
      const std::vector<bool> p_fix,
      Eigen::VectorXi& p_set,
      Eigen::VectorXd& kn,
      const int N,
      Eigen::VectorXd& angles,
      Eigen::VectorXd& S
    );


    IGL_INLINE void nrosy(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::VectorXi& b,
      const Eigen::VectorXd& br,
      const std::vector<Eigen::MatrixXd>& TP_set,
      Eigen::VectorXi& pj,
      Eigen::VectorXd& kn,
      const int N,
      const std::vector<std::vector<std::pair<int,int>>>& coeff,
      const Eigen::VectorXd& rhs,
      Eigen::VectorXd& angles,
      Eigen::VectorXd& S
    );
  }
}
}

#ifndef IGL_STATIC_LIBRARY
#  include "nrosy.cpp"
#endif

#endif
