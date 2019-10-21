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
    
    IGL_INLINE void initTP(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,std::vector<Eigen::MatrixXd>& TP_set);
    
    class NRosyField
    {
    public:
      // Init
      IGL_INLINE NRosyField(const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F);

      // Generate the N-rosy field
      // N degree of the rosy field
      // round separately: round the integer variables one at a time, slower but higher quality
      IGL_INLINE void solve(const Eigen::VectorXi& p_set, 
                            const std::vector<bool>& p_fix,
                            const int N,
                            const Eigen::VectorXd& pj_rhs,
                            const bool rounding);

      // Set a hard constraint on fid
      // fid: face id
      // v: direction to fix (in 3d)
      IGL_INLINE void setConstraintHard(int fid, const Eigen::Vector3d& v);

      // Set a soft constraint on fid
      // fid: face id
      // w: weight of the soft constraint, clipped between 0 and 1
      // v: direction to fix (in 3d)
      IGL_INLINE void setConstraintSoft(int fid, double w, const Eigen::Vector3d& v);

      // Set the ratio between smoothness and soft constraints (0 -> smoothness only, 1 -> soft constr only)
      IGL_INLINE void setSoftAlpha(double alpha);

      // Reset constraints (at least one constraint must be present or solve will fail)
      IGL_INLINE void resetConstraints();

      // Return the current field
      IGL_INLINE Eigen::MatrixXd getFieldPerFace();
      
      IGL_INLINE Eigen::MatrixXd getFFieldPerFace();

      // Compute singularity indexes
      IGL_INLINE void findCones(int N);

      // Return the singularities
      IGL_INLINE Eigen::VectorXd getSingularityIndexPerVertex();
      
      IGL_INLINE Eigen::VectorXd getKappa();

      IGL_INLINE Eigen::VectorXi getPeriodJump();
      
      // build constraints on period jumps
      IGL_INLINE void pjconstraints();
      IGL_INLINE Eigen::VectorXd getPJRhs();
      IGL_INLINE void setPJRhs(const Eigen::VectorXd& pj_rhs);
      
      // load kappa
      IGL_INLINE void loadk(const Eigen::VectorXd& kn);
      
      // set TP
      IGL_INLINE void setTP(const std::vector<Eigen::MatrixXd>& TP_set);

      // set bds
      IGL_INLINE void setbds();
      
      // setup seam info
      IGL_INLINE void set_seams(const Eigen::MatrixXi& mask){ is_seam = mask; };
      IGL_INLINE void set_mt(const Eigen::MatrixXi& tt, const Eigen::MatrixXi& tti){mt = tt; mti = tti;};

    private:
      // Compute angle differences between reference frames
      IGL_INLINE void computek();

      // Remove useless matchings
      IGL_INLINE void reduceSpace();

      // Prepare the system matrix
      IGL_INLINE void prepareSystemMatrix(int N);

      // Solve with roundings using CoMIso
      IGL_INLINE void solveRoundings();
      
      IGL_INLINE void solveNoRoundings();
      
      IGL_INLINE void roundAndFix();
      IGL_INLINE void roundAndFixToZero();

      // Convert a vector in 3d to an angle wrt the local reference system
      IGL_INLINE double convert3DtoLocal(unsigned fid, const Eigen::Vector3d& v);

      // Convert an angle wrt the local reference system to a 3d vector
      IGL_INLINE Eigen::Vector3d convertLocalto3D(unsigned fid, double a);

      // Compute the per vertex angle defect
      IGL_INLINE Eigen::VectorXd angleDefect();

      // Temporary variable for the field
      Eigen::VectorXd angles;

      // Hard constraints
      Eigen::VectorXd hard;
      std::vector<bool> isHard;
      
      // linear system built on period jump constraints
      Eigen::SparseMatrix<double> C;
      Eigen::VectorXd C_rhs;

      // Soft constraints
      Eigen::VectorXd soft;
      Eigen::VectorXd wSoft;
      double softAlpha;

      // Face Topology
      Eigen::MatrixXi TT, TTi;

      // Edge Topology
      Eigen::MatrixXi EV, FE, EF;
      std::vector<bool> isBorderEdge;
      std::vector<bool> is_bd;
      int n_no_bd;

      // Per Edge information
      // Angle between two reference frames
      Eigen::VectorXd k;

      // Jumps
      Eigen::VectorXi p;
      std::vector<bool> pFixed;
      Eigen::VectorXi to_no_bd;

      // Mesh
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;

      // Normals per face
      Eigen::MatrixXd N;

      // Singularity index
      Eigen::VectorXd singularityIndex;

      // Reference frame per triangle
      std::vector<Eigen::MatrixXd> TPs;

      // System stuff
      Eigen::SparseMatrix<double> A;
      Eigen::VectorXd b;
      Eigen::VectorXi tag_t;
      Eigen::VectorXi tag_p;
      Eigen::VectorXd sol;
      
      // Extension data
      Eigen::MatrixXi is_seam; // dim (nf x 3)
      Eigen::MatrixXi mt;      // match seams
      Eigen::MatrixXi mti;     // adjacent triangles info
    };
    
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
      
    IGL_INLINE void nrosy(
      const Eigen::MatrixXd& V,
      const Eigen::MatrixXi& F,
      const Eigen::VectorXi& b,
      const Eigen::MatrixXd& bc,
      const std::vector<Eigen::MatrixXd>& TP_set,
      Eigen::VectorXi& p_set,
      const std::vector<bool>& p_fix,
      const Eigen::VectorXd& kn,
      const int N,
      const Eigen::VectorXd& pj_rhs,
      const bool pj_constraints,
      Eigen::MatrixXd& R,
      Eigen::VectorXd& S
    );
    
  }
}
}

#ifndef IGL_STATIC_LIBRARY
#  include "nrosy.cpp"
#endif

#endif
