// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "nrosy.h"

#include <igl/copyleft/comiso/nrosy.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/edge_topology.h>
#include <igl/per_face_normals.h>

#include <iostream>
#include <fstream>

#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <queue>

#include <gmm/gmm.h>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/MISolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <igl/PI.h>
#include <igl/boundary_loop.h>

void igl::copyleft::comiso::initTP(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  std::vector<Eigen::MatrixXd>& TP_set
){
  using namespace Eigen;
  Eigen::MatrixXd N;
  // Generate normals per face
  igl::per_face_normals(V, F, N);
  TP_set.clear();
  // Generate reference frames
  for(unsigned fid=0; fid<F.rows(); ++fid)
  {
    // First edge
    Vector3d e1 = V.row(F(fid,1)) - V.row(F(fid,0));
    e1.normalize();
    Vector3d e2 = N.row(fid);
    e2 = e2.cross(e1);
    e2.normalize();

    MatrixXd TP(2,3);
    TP << e1.transpose(), e2.transpose();
    TP_set.push_back(TP);
  }
}

void igl::copyleft::comiso::NRosyField::loadk(const Eigen::VectorXd& kn){
  k = kn;
}

void igl::copyleft::comiso::NRosyField::setTP(const std::vector<Eigen::MatrixXd>& TP_set){
  TPs = TP_set;
}

igl::copyleft::comiso::NRosyField::NRosyField(const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F)
{
  using namespace std;
  using namespace Eigen;

  V = _V;
  F = _F;

  assert(V.rows() > 0);
  assert(F.rows() > 0);


  // Generate topological relations
  igl::triangle_triangle_adjacency(F,TT,TTi);
  igl::edge_topology(V,F, EV, FE, EF);

  // Flag border edges
  isBorderEdge.resize(EV.rows());
  for(unsigned i=0; i<EV.rows(); ++i)
    isBorderEdge[i] = (EF(i,0) == -1) || ((EF(i,1) == -1));

  // Generate normals per face
  igl::per_face_normals(V, F, N);

  // Generate reference frames
  // for(unsigned fid=0; fid<F.rows(); ++fid)
  // {
  //   // First edge
  //   Vector3d e1 = V.row(F(fid,1)) - V.row(F(fid,0));
  //   e1.normalize();
  //   Vector3d e2 = N.row(fid);
  //   e2 = e2.cross(e1);
  //   e2.normalize();

  //   MatrixXd TP(2,3);
  //   TP << e1.transpose(), e2.transpose();
  //   TPs.push_back(TP);
  // }

  // Alloc internal variables
  angles = VectorXd::Zero(F.rows());
  p = VectorXi::Zero(EV.rows());
  pFixed.resize(EV.rows());
  k = VectorXd::Zero(EV.rows());
  singularityIndex = VectorXd::Zero(V.rows());

  // Reset the constraints
  resetConstraints();

  // Compute k, differences between reference frames
  // computek();

  softAlpha = 0.5;
}

void igl::copyleft::comiso::NRosyField::setSoftAlpha(double alpha)
{
  assert(alpha >= 0 && alpha < 1);
  softAlpha = alpha;
}


void igl::copyleft::comiso::NRosyField::prepareSystemMatrix(const int N)
{
  using namespace std;
  using namespace Eigen;

  double Nd = N;

  // Minimize the MIQ energy
  // Energy on edge ij is
  //     (t_i - t_j + kij + pij*(2*pi/N))^2
  // Partial derivatives:
  //   t_i: 2     ( t_i - t_j + kij + pij*(2*pi/N)) = 0
  //   t_j: 2     (-t_i + t_j - kij - pij*(2*pi/N)) = 0
  //   pij: 4pi/N ( t_i - t_j + kij + pij*(2*pi/N)) = 0
  //
  //          t_i      t_j         pij       kij
  // t_i [     2       -2           4pi/N      2    ]
  // t_j [    -2        2          -4pi/N     -2    ]
  // pij [   4pi/N   -4pi/N    2*(2pi/N)^2   4pi/N  ]

  // Count and tag the variables
  tag_t = VectorXi::Constant(F.rows(),-1);
  vector<int> id_t;
  int count = 0;
  for(unsigned i=0; i<F.rows(); ++i)
    if (!isHard[i])
    {
      tag_t(i) = count++;
      id_t.push_back(i);
    }

  unsigned count_t = id_t.size();
  tag_p = VectorXi::Constant(EF.rows(),-1);
  vector<int> id_p;
  for(unsigned i=0; i<EF.rows(); ++i)
  {
    if (!pFixed[i])
    {
      // if it is not fixed then it is a variable
      tag_p(i) = count++;
    }

    // if it is not a border edge,
    if (!isBorderEdge[i])
    {
      // and it is not between two fixed faces
      if (!(isHard[EF(i,0)] && isHard[EF(i,1)]))
      {
          // then it participates in the energy!
          id_p.push_back(i);
      }
    }
  }
  unsigned count_p = count - count_t;
  // System sizes: A (count_t + count_p) x (count_t + count_p)
  //               b (count_t + count_p)

  b = VectorXd::Zero(count_t + count_p);

  std::vector<Eigen::Triplet<double> > T;
  T.reserve(3 * 4 * count_p);
  for(unsigned r=0; r<id_p.size(); ++r)
  {
    int eid = id_p[r];
    int i = EF(eid,0);
    int j = EF(eid,1);
    bool isFixed_i = isHard[i];
    bool isFixed_j = isHard[j];
    bool isFixed_p = pFixed[eid];
    int row;
    // (i)-th row: t_i [     2       -2           4pi/N      2    ]
    if (!isFixed_i)
    {
      row = tag_t[i];
      if (isFixed_i) b(row) += -2               * hard[i]; else T.push_back(Eigen::Triplet<double>(row,tag_t[i]  , 2             ));
      if (isFixed_j) b(row) +=  2               * hard[j]; else T.push_back(Eigen::Triplet<double>(row,tag_t[j]  ,-2             ));
      if (isFixed_p) b(row) += -((4 * igl::PI)/Nd) * p[eid] ; else T.push_back(Eigen::Triplet<double>(row,tag_p[eid],((4 * igl::PI)/Nd)));
      b(row) += -2 * k[eid];
      assert(hard[i] == hard[i]);
      assert(hard[j] == hard[j]);
      assert(p[eid] == p[eid]);
      assert(k[eid] == k[eid]);
      assert(b(row) == b(row));
    }
    // (j)+1 -th row: t_j [    -2        2          -4pi/N     -2    ]
    if (!isFixed_j)
    {
      row = tag_t[j];
      if (isFixed_i) b(row) += 2               * hard[i]; else T.push_back(Eigen::Triplet<double>(row,tag_t[i]  , -2             ));
      if (isFixed_j) b(row) += -2              * hard[j]; else T.push_back(Eigen::Triplet<double>(row,tag_t[j] ,  2              ));
      if (isFixed_p) b(row) += ((4 * igl::PI)/Nd) * p[eid] ; else T.push_back(Eigen::Triplet<double>(row,tag_p[eid],-((4 * igl::PI)/Nd)));
      b(row) += 2 * k[eid];
      assert(k[eid] == k[eid]);
      assert(b(row) == b(row));
    }
    // (r*3)+2 -th row: pij [   4pi/N   -4pi/N    2*(2pi/N)^2   4pi/N  ]
    if (!isFixed_p)
    {
      row = tag_p[eid];
      if (isFixed_i) b(row) += -(4 * igl::PI)/Nd              * hard[i]; else T.push_back(Eigen::Triplet<double>(row,tag_t[i] ,   (4 * igl::PI)/Nd             ));
      if (isFixed_j) b(row) +=  (4 * igl::PI)/Nd              * hard[j]; else T.push_back(Eigen::Triplet<double>(row,tag_t[j] ,  -(4 * igl::PI)/Nd             ));
      if (isFixed_p) b(row) += -(2 * pow(((2*igl::PI)/Nd),2)) * p[eid] ;  else T.push_back(Eigen::Triplet<double>(row,tag_p[eid],  (2 * pow(((2*igl::PI)/Nd),2))));
      b(row) += - (4 * igl::PI)/Nd * k[eid];
      assert(k[eid] == k[eid]);
      assert(b(row) == b(row));
    }

  }
  A = SparseMatrix<double>(count_t + count_p, count_t + count_p);
  A.setFromTriplets(T.begin(), T.end());
  // Soft constraints
  bool addSoft = false;

  for(unsigned i=0; i<wSoft.size();++i)
    if (wSoft[i] != 0)
      addSoft = true;

  if (addSoft)
  {
    cerr << " Adding soft here: " << endl;
    cerr << " softAplha: " << softAlpha << endl;
    VectorXd bSoft = VectorXd::Zero(count_t + count_p);

    std::vector<Eigen::Triplet<double> > TSoft;
    TSoft.reserve(2 * count_p);

    for(unsigned i=0; i<F.rows(); ++i)
    {
      int varid = tag_t[i];
      if (varid != -1) // if it is a variable in the system
      {
        TSoft.push_back(Eigen::Triplet<double>(varid,varid,wSoft[i]));
        bSoft[varid] += wSoft[i] * soft[i];
      }
    }
    SparseMatrix<double> ASoft(count_t + count_p, count_t + count_p);
    ASoft.setFromTriplets(TSoft.begin(), TSoft.end());

//    ofstream s("/Users/daniele/As.txt");
//    for(unsigned i=0; i<TSoft.size(); ++i)
//      s << TSoft[i].row() << " " << TSoft[i].col() << " " << TSoft[i].value() << endl;
//    s.close();

//    ofstream s2("/Users/daniele/bs.txt");
//    for(unsigned i=0; i<bSoft.rows(); ++i)
//      s2 << bSoft(i) << endl;
//    s2.close();

    // Stupid Eigen bug
    SparseMatrix<double> Atmp (count_t + count_p, count_t + count_p);
    SparseMatrix<double> Atmp2(count_t + count_p, count_t + count_p);
    SparseMatrix<double> Atmp3(count_t + count_p, count_t + count_p);

    // Merge the two part of the energy
    Atmp = (1.0 - softAlpha)*A;
    Atmp2 = softAlpha * ASoft;
    Atmp3 = Atmp+Atmp2;

    A = Atmp3;
    b = b*(1.0 - softAlpha) + bSoft * softAlpha;
  }

//  ofstream s("/Users/daniele/A.txt");
//  for (int k=0; k<A.outerSize(); ++k)
//    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
//    {
//      s << it.row() << " " << it.col() << " " << it.value() << endl;
//    }
//  s.close();
//
//  ofstream s2("/Users/daniele/b.txt");
//  for(unsigned i=0; i<b.rows(); ++i)
//    s2 << b(i) << endl;
//  s2.close();
}

void igl::copyleft::comiso::NRosyField::solveNoRoundings()
{
  using namespace std;
  using namespace Eigen;

  // Solve the linear system
  SimplicialLDLT<SparseMatrix<double> > solver;
  solver.compute(A);
  VectorXd x = solver.solve(b);
  // Copy the result back
  for(unsigned i=0; i<F.rows(); ++i)
    if (tag_t[i] != -1)
      angles[i] = x(tag_t[i]);
    else
      angles[i] = hard[i];

  for(unsigned i=0; i<EF.rows(); ++i)
    if(tag_p[i]  != -1)
      p[i] = roundl(x[tag_p[i]]);
}

/// function to print the equations corresponding to the matrices of an equation system
template<class MatrixT>
void print_equations( const MatrixT& _B)
{
  int m = gmm::mat_nrows( _B);
  int n = gmm::mat_ncols( _B);
  for( int i = 0; i < m; ++i)
  {
    for( int j = 0; j < n-1; ++j)
    {
      if( _B(i,j) != 0.0)
        std::cout << _B(i,j) << "*x" << j;
      else
        std::cout << "   0 ";
      if( j < n-2 ) std::cout << " + ";
    }
    std::cout << " = " << _B(i, n-1) << std::endl;
  }
}

// set up the linear constraints for period jumps
void igl::copyleft::comiso::NRosyField::pjconstraints(){
  
  using namespace std;
  using namespace Eigen;
  
  // sum \pij = val
  unsigned int n_var = A.rows();
  int n_constr = 0;
  Eigen::VectorXi bd;
  igl::boundary_loop(F,bd);
  std::vector<bool> is_bd(V.rows(),false);
  for(int id=0;id<bd.rows();id++)
    is_bd[bd(id)] = true;
  to_no_bd.resize(V.rows());
  to_no_bd.setConstant(-1);
  int count = 0;
  for(int i=0;i<V.rows();i++){
    if(!is_bd[i]){
      to_no_bd(i) = count;
      count++;
    }
  }
  std::vector<Triplet<double>> T;
  for(unsigned i=0;i<EV.rows();i++){
    int xid = tag_p[i];
    if(xid != -1){
      if(to_no_bd(EV(i,0))!=-1)
        T.push_back(Triplet<double>(to_no_bd(EV(i,0)), xid, 1));
      if(to_no_bd(EV(i,1))!=-1)
        T.push_back(Triplet<double>(to_no_bd(EV(i,1)), xid, -1));
    }
  }
  C = SparseMatrix<double>(count, n_var);
  C.setFromTriplets(T.begin(), T.end());
  
  
}
extern int pick;
void igl::copyleft::comiso::NRosyField::solveRoundings()
{
  using namespace std;
  using namespace Eigen;

  unsigned n = A.rows();

  gmm::col_matrix< gmm::wsvector< double > > gmm_A;
  std::vector<double> gmm_b;
  std::vector<int> ids_to_round;
  std::vector<double> x;

  gmm_A.resize(n,n);
  gmm_b.resize(n);
  x.resize(n);

  // Copy A
  for (int k=0; k<A.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    {
      gmm_A(it.row(),it.col()) += it.value();
    }

  // Copy b
  for(unsigned i=0; i<n;++i)
    gmm_b[i] = b[i];

  // Set variables to round
  ids_to_round.clear();
  for(unsigned i=0; i<tag_p.size();++i)
    if(tag_p[i] != -1)
      ids_to_round.push_back(tag_p[i]);

  // Empty constraints
  int n_constr = 0;
  gmm::row_matrix< gmm::wsvector< double > > gmm_C(C.rows(), n+1);
  if(C.rows() > 0 && C_rhs.rows() > 0){
    for(int k=0;k < C.outerSize();k++){
      for(SparseMatrix<double>::InnerIterator it(C,k); it; ++it){
        gmm_C(it.row(),it.col()) += it.value();
      }
    }
    for(int i=0;i<tag_p.size();i++){
      int a = EV(i,0);
      int b = EV(i,1);
      if(tag_p(i) == -1){ // fixed pij, then remove it from C_rhs
        if(to_no_bd(a) != -1)
          C_rhs(to_no_bd(a)) -= p(i);
        if(to_no_bd(b) != -1)
          C_rhs(to_no_bd(b)) += p(i);
      }
    }
    for(int i=0;i<C.rows();i++){
      gmm_C(i, n) = -C_rhs(i);
    }
  }

  COMISO::ConstrainedSolver cs;
  //print_miso_settings(cs.misolver());
  cs.solve(gmm_C, gmm_A, x, gmm_b, ids_to_round, 0.0, false, true);
  if(C.rows() > 0){
    Eigen::VectorXd sol(x.size());
    for(int i=0;i<sol.rows();i++)
      sol(i) = x[i];
    if((C*sol-C_rhs).norm() > 1e-10){
      std::cout<<C*sol-C_rhs<<std::endl;
    }else
      std::cout<<"residual = "<<(C*sol-C_rhs).norm()<<std::endl;
  }
  
  // Copy the result back
  for(unsigned i=0; i<F.rows(); ++i)
    if (tag_t[i] != -1)
      angles[i] = x[tag_t[i]];
    else
      angles[i] = hard[i];

  for(unsigned i=0; i<EF.rows(); ++i)
    if(tag_p[i]  != -1)
      p[i] = roundl(x[tag_p[i]]);
}


void igl::copyleft::comiso::NRosyField::roundAndFix()
{
  for(unsigned i=0; i<p.rows(); ++i)
    pFixed[i] = true;
}

void igl::copyleft::comiso::NRosyField::roundAndFixToZero()
{
  for(unsigned i=0; i<p.rows(); ++i)
  {
    pFixed[i] = true;
    p[i] = 0;
  }
}

void igl::copyleft::comiso::NRosyField::solve(const Eigen::VectorXi& p_set, 
                                              const std::vector<bool>& p_fix,
                                              const int N,
                                              const Eigen::VectorXd& pj_rhs,
                                              const bool with_pj_constraints)
{
  // Reduce the search space by fixing matchings
  bool fixed = false;
  pFixed = p_fix;
  for(int i=0;i<pFixed.size();i++){
    if(pFixed[i])
      fixed = true;
  }
  if(!fixed)
    reduceSpace();
  else
    p = p_set;
  
  // Build the system
  prepareSystemMatrix(N);

  // Solve with integer roundings
  if(with_pj_constraints){
    pjconstraints();
    setPJRhs(pj_rhs);
  }
  
  solveRoundings();
  
  // This is a very greedy solving strategy
  // // Solve with no roundings
  // solveNoRoundings();
  //
  // // Round all p and fix them
  // roundAndFix();
  //
  // // Build the system
  // prepareSystemMatrix(N);
  //
  // // Solve with no roundings (they are all fixed)
  // solveNoRoundings();

  // Find the cones
  findCones(N);
}

void igl::copyleft::comiso::NRosyField::setConstraintHard(const int fid, const Eigen::Vector3d& v)
{
  isHard[fid] = true;
  hard(fid) = convert3DtoLocal(fid, v);
}

// void igl::copyleft::comiso::NRosyField::setSingularityConstraints(
//   const Eigen::VectorXd& S
// ){
//   std::cout << " Adding soft here: " << std::endl;
//   // std::cout << " sing_alpha: " << sing_alpha << std::endl;
//   Eigen::VectorXd b_sing = VectorXd::Zero();

//     std::vector<Eigen::Triplet<double> > TSoft;
//     TSoft.reserve(2 * count_p);

//     for(unsigned i=0; i<F.rows(); ++i)
//     {
//       int varid = tag_t[i];
//       if (varid != -1) // if it is a variable in the system
//       {
//         TSoft.push_back(Eigen::Triplet<double>(varid,varid,wSoft[i]));
//         bSoft[varid] += wSoft[i] * soft[i];
//       }
//     }
//     SparseMatrix<double> ASoft(count_t + count_p, count_t + count_p);
//     ASoft.setFromTriplets(TSoft.begin(), TSoft.end());

// //    ofstream s("/Users/daniele/As.txt");
// //    for(unsigned i=0; i<TSoft.size(); ++i)
// //      s << TSoft[i].row() << " " << TSoft[i].col() << " " << TSoft[i].value() << endl;
// //    s.close();

// //    ofstream s2("/Users/daniele/bs.txt");
// //    for(unsigned i=0; i<bSoft.rows(); ++i)
// //      s2 << bSoft(i) << endl;
// //    s2.close();

//     // Stupid Eigen bug
//     SparseMatrix<double> Atmp (count_t + count_p, count_t + count_p);
//     SparseMatrix<double> Atmp2(count_t + count_p, count_t + count_p);
//     SparseMatrix<double> Atmp3(count_t + count_p, count_t + count_p);

//     // Merge the two part of the energy
//     Atmp = (1.0 - softAlpha)*A;
//     Atmp2 = softAlpha * ASoft;
//     Atmp3 = Atmp+Atmp2;

//     A = Atmp3;
//     b = b*(1.0 - softAlpha) + bSoft * softAlpha;
// }

void igl::copyleft::comiso::NRosyField::setConstraintSoft(const int fid, const double w, const Eigen::Vector3d& v)
{
  wSoft(fid) = w;
  soft(fid) = convert3DtoLocal(fid, v);
}

void igl::copyleft::comiso::NRosyField::resetConstraints()
{
  using namespace std;
  using namespace Eigen;

  isHard.resize(F.rows());
  for(unsigned i=0; i<F.rows(); ++i)
    isHard[i] = false;
  hard   = VectorXd::Zero(F.rows());

  wSoft  = VectorXd::Zero(F.rows());
  soft   = VectorXd::Zero(F.rows());
}

Eigen::MatrixXd igl::copyleft::comiso::NRosyField::getFieldPerFace()
{
  using namespace std;
  using namespace Eigen;

  MatrixXd result(F.rows(),3);
  for(unsigned i=0; i<F.rows(); ++i)
    result.row(i) = convertLocalto3D(i, angles(i));
  return result;
}

Eigen::MatrixXd igl::copyleft::comiso::NRosyField::getFFieldPerFace()
{
  using namespace std;
  using namespace Eigen;

  MatrixXd result(F.rows(),6);
  for(unsigned i=0; i<F.rows(); ++i)
  {
      Vector3d v1 = convertLocalto3D(i, angles(i));
      Vector3d n = N.row(i);
      Vector3d v2 = n.cross(v1);
      v1.normalize();
      v2.normalize();

      result.block(i,0,1,3) = v1.transpose();
      result.block(i,3,1,3) = v2.transpose();
  }
  return result;
}


void igl::copyleft::comiso::NRosyField::computek()
{
  // For every non-border edge
  for (unsigned eid = 0; eid < EF.rows(); ++eid)
  {
    if (!isBorderEdge[eid])
    {
      int fid0 = EF(eid,0);
      int fid1 = EF(eid,1);

      Eigen::Vector3d N0 = N.row(fid0);
      Eigen::Vector3d N1 = N.row(fid1);

      // find common edge on triangle 0 and 1
      int fid0_vc = -1;
      int fid1_vc = -1;
      for (unsigned i=0;i<3;++i)
      {
        if (EV(eid,0) == F(fid0,i))
          fid0_vc = i;
        if (EV(eid,1) == F(fid1,i))
          fid1_vc = i;
      }
      assert(fid0_vc != -1);
      assert(fid1_vc != -1);

      Eigen::Vector3d common_edge = V.row(F(fid0,(fid0_vc+1)%3)) - V.row(F(fid0,fid0_vc));
      common_edge.normalize();

      // Map the two triangles in a new space where the common edge is the x axis and the N0 the z axis
      Eigen::MatrixXd P(3,3);
      Eigen::VectorXd o = V.row(F(fid0,fid0_vc));
      Eigen::VectorXd tmp = N0.cross(common_edge);
      P << common_edge, tmp, N0;
      P.transposeInPlace();


      Eigen::MatrixXd V0(3,3);
      V0.row(0) = V.row(F(fid0,0)).transpose() -o;
      V0.row(1) = V.row(F(fid0,1)).transpose() -o;
      V0.row(2) = V.row(F(fid0,2)).transpose() -o;

      V0 = (P*V0.transpose()).transpose();

      assert(V0(0,2) < 1e-10);
      assert(V0(1,2) < 1e-10);
      assert(V0(2,2) < 1e-10);

      Eigen::MatrixXd V1(3,3);
      V1.row(0) = V.row(F(fid1,0)).transpose() -o;
      V1.row(1) = V.row(F(fid1,1)).transpose() -o;
      V1.row(2) = V.row(F(fid1,2)).transpose() -o;
      V1 = (P*V1.transpose()).transpose();

      assert(V1(fid1_vc,2) < 1e-10);
      assert(V1((fid1_vc+1)%3,2) < 1e-10);

      // compute rotation R such that R * N1 = N0
      // i.e. map both triangles to the same plane
      double alpha = -std::atan2(-V1((fid1_vc + 2) % 3, 2), -V1((fid1_vc + 2) % 3, 1));

      Eigen::MatrixXd R(3,3);
      R << 1,          0,            0,
           0, std::cos(alpha), -std::sin(alpha),
           0, std::sin(alpha),  std::cos(alpha);
      V1 = (R*V1.transpose()).transpose();

      assert(V1(0,2) < 1e-10);
      assert(V1(1,2) < 1e-10);
      assert(V1(2,2) < 1e-10);

      // measure the angle between the reference frames
      // k_ij is the angle between the triangle on the left and the one on the right
      Eigen::VectorXd ref0 = V0.row(1) - V0.row(0);
      Eigen::VectorXd ref1 = V1.row(1) - V1.row(0);

      ref0.normalize();
      ref1.normalize();
      
      double ktemp = - std::atan2(ref1(1), ref1(0)) + std::atan2(ref0(1), ref0(0));
      
      // make sure kappa is in corret range 
      auto pos_fmod = [](double x, double y){
        return (0 == y) ? x : x - y * floor(x/y);
      };
      ktemp = pos_fmod(ktemp, 2*igl::PI);
      if (ktemp > igl::PI) 
        ktemp -= 2*igl::PI;
      
      // just to be sure, rotate ref0 using angle ktemp...
      Eigen::MatrixXd R2(2,2);
      R2 << std::cos(-ktemp), -std::sin(-ktemp), std::sin(-ktemp), std::cos(-ktemp);

      tmp = R2*ref0.head<2>();

      assert(tmp(0) - ref1(0) < 1e-10);
      assert(tmp(1) - ref1(1) < 1e-10);

      k[eid] = ktemp;
    }
  }

}


void igl::copyleft::comiso::NRosyField::reduceSpace()
{
  using namespace std;
  using namespace Eigen;

  // All variables are free in the beginning
  for(unsigned i=0; i<EV.rows(); ++i)
    pFixed[i] = false;

  vector<VectorXd> debug;

  // debug
//  MatrixXd B(F.rows(),3);
//  for(unsigned i=0; i<F.rows(); ++i)
//    B.row(i) = 1./3. * (V.row(F(i,0)) + V.row(F(i,1)) + V.row(F(i,2)));

  vector<bool> visited(EV.rows());
  for(unsigned i=0; i<EV.rows(); ++i)
    visited[i] = false;

  vector<bool> starting(EV.rows());
  for(unsigned i=0; i<EV.rows(); ++i)
    starting[i] = false;

  queue<int> q;
  for(unsigned i=0; i<F.rows(); ++i)
    if (isHard[i] || wSoft[i] != 0)
    {
      q.push(i);
      starting[i] = true;
    }

  // Reduce the search space (see MI paper)
  while (!q.empty())
  {
    int c = q.front();
    q.pop();

    visited[c] = true;
    for(int i=0; i<3; ++i)
    {
      int eid = FE(c,i);
      int fid = TT(c,i);

      // skip borders
      if (fid != -1)
      {
        assert((EF(eid,0) == c && EF(eid,1) == fid) || (EF(eid,1) == c && EF(eid,0) == fid));
        // for every neighbouring face
        if (!visited[fid] && !starting[fid])
        {
          pFixed[eid] = true;
          p[eid] = 0;
          visited[fid] = true;
          q.push(fid);

        }
      }
      else
      {
        // fix borders
        pFixed[eid] = true;
        p[eid] = 0;
      }
    }

  }

  // Force matchings between fixed faces
  for(unsigned i=0; i<F.rows();++i)
  {
    if (isHard[i])
    {
      for(unsigned int j=0; j<3; ++j)
      {
        int fid = TT(i,j);
        if ((fid!=-1) && (isHard[fid]))
        {
          // i and fid are adjacent and fixed
          int eid = FE(i,j);
          int fid0 = EF(eid,0);
          int fid1 = EF(eid,1);

          pFixed[eid] = true;
          p[eid] = roundl(2.0/igl::PI*(hard(fid1) - hard(fid0) - k(eid)));
        }
      }
    }
  }

//  std::ofstream s("/Users/daniele/debug.txt");
//  for(unsigned i=0; i<debug.size(); i += 2)
//    s << debug[i].transpose() << " " << debug[i+1].transpose() << endl;
//  s.close();

}

double igl::copyleft::comiso::NRosyField::convert3DtoLocal(unsigned fid, const Eigen::Vector3d& v)
{
  using namespace std;
  using namespace Eigen;

  // Project onto the tangent plane
  Vector2d vp = TPs[fid] * v;

  // Convert to angle
  return atan2(vp(1),vp(0));
}

Eigen::Vector3d igl::copyleft::comiso::NRosyField::convertLocalto3D(unsigned fid, double a)
{
  using namespace std;
  using namespace Eigen;
  Vector2d vp(cos(a),sin(a));
  
  return vp.transpose() * TPs[fid];
}

Eigen::VectorXd igl::copyleft::comiso::NRosyField::angleDefect()
{
  Eigen::VectorXd A = Eigen::VectorXd::Constant(V.rows(),2*igl::PI);

  for (unsigned i=0; i < F.rows(); ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      Eigen::VectorXd a = V.row(F(i,(j+1)%3)) - V.row(F(i,j));
      Eigen::VectorXd b = V.row(F(i,(j+2)%3)) - V.row(F(i,j));
      double t = a.transpose()*b;
      t /= (a.norm() * b.norm());
      A(F(i,j)) -= acos(t);
    }
  }

  return A;
}

// setter
void igl::copyleft::comiso::NRosyField::setPJRhs(const Eigen::VectorXd& pj_rhs){
  C_rhs = pj_rhs;
  
  Eigen::VectorXi bd;
  igl::boundary_loop(F,bd);
  std::vector<bool> is_bd(V.rows(),false);
  for(int id=0;id<bd.rows();id++)
    is_bd[bd(id)] = true;
  Eigen::VectorXi to_no_bd(V.rows()); 
  to_no_bd.setConstant(-1);
  int count = 0;
  for(int i=0;i<V.rows();i++){
    if(!is_bd[i]){
      to_no_bd(i) = count;
      count++;
    }
  }
  Eigen::VectorXd C_rhs_nobd(count);
  int r = 0;
  if(count == C_rhs.rows()) return;
  for(int i=0;i<C_rhs.rows();i++){
    if(!is_bd[i]){
      C_rhs_nobd(r) = C_rhs(i);
      r++;
    }
  }
  C_rhs = C_rhs_nobd;
}

// getter
Eigen::VectorXd igl::copyleft::comiso::NRosyField::getPJRhs(){
  findCones(4);
  // std::cout<<C_rhs<<std::endl;
  Eigen::VectorXi bd;
  igl::boundary_loop(F,bd);
  std::vector<bool> is_bd(V.rows(),false);
  for(int id=0;id<bd.rows();id++)
    is_bd[bd(id)] = true;
  Eigen::VectorXi to_no_bd(V.rows());
  to_no_bd.setConstant(-1);
  int count = 0;
  for(int i=0;i<V.rows();i++){
    if(!is_bd[i]){
      to_no_bd(i) = count;
      count++;
    }
  }
  Eigen::VectorXd C_rhs_nobd(count);
  int r = 0;
  for(int i=0;i<C_rhs.rows();i++){
    if(!is_bd[i]){
      C_rhs_nobd(r) = C_rhs(i);
      r++;
    }
  }
  return C_rhs_nobd;
}

void igl::copyleft::comiso::NRosyField::findCones(int N)
{
  // Compute I0, see http://www.graphics.rwth-aachen.de/media/papers/bommes_zimmer_2009_siggraph_011.pdf for details

  Eigen::VectorXd I0 = Eigen::VectorXd::Zero(V.rows());

  // first the k
  for (unsigned i=0; i < EV.rows(); ++i)
  {
    if (!isBorderEdge[i])
    {
      I0(EV(i,0)) += k(i);
      I0(EV(i,1)) -= k(i);
    }
  }

  // then the A
  Eigen::VectorXd A = angleDefect();

  I0 = I0 + A;

  // normalize
  I0 = I0 / (2*igl::PI);
  // round to integer (remove numerical noise)
  for (unsigned i=0; i < I0.size(); ++i)
    I0(i) = round(I0(i));
  // compute I
  Eigen::VectorXd I = I0;
  C_rhs.setZero(I.rows());
  for (unsigned i=0; i < EV.rows(); ++i)
  {
    if (!isBorderEdge[i])
    {
      I(EV(i,0)) += double(p(i))/double(N);
      I(EV(i,1)) -= double(p(i))/double(N);
    }
  }

  // Clear the vertices on the edges
  for (unsigned i=0; i < EV.rows(); ++i)
  {
    if (isBorderEdge[i])
    {
      I0(EV(i,0)) = 0;
      I0(EV(i,1)) = 0;
      I(EV(i,0)) = 0;
      I(EV(i,1)) = 0;
      A(EV(i,0)) = 0;
      A(EV(i,1)) = 0;
    }
  }

  singularityIndex = I;
  C_rhs = 4*(I-I0);

}

Eigen::VectorXd igl::copyleft::comiso::NRosyField::getSingularityIndexPerVertex()
{
  return singularityIndex;
}

Eigen::VectorXd igl::copyleft::comiso::NRosyField::getKappa()
{
  return k;
}

Eigen::VectorXi igl::copyleft::comiso::NRosyField::getPeriodJump()
{
  return p;
}



IGL_INLINE void igl::copyleft::comiso::nrosy(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& b,
  const Eigen::MatrixXd& bc,
  const Eigen::VectorXi& b_soft,
  const Eigen::VectorXd& w_soft,
  const Eigen::MatrixXd& bc_soft,
  const int N,
  const double soft,
  Eigen::MatrixXd& R,
  Eigen::VectorXd& S
  )
{
  // Init solver
  igl::copyleft::comiso::NRosyField solver(V,F);

  // Add hard constraints
  for (unsigned i=0; i<b.size();++i)
    solver.setConstraintHard(b(i),bc.row(i));

  // Add soft constraints
  for (unsigned i=0; i<b_soft.size();++i)
    solver.setConstraintSoft(b_soft(i),w_soft(i),bc_soft.row(i));

  // Set the soft constraints global weight
  solver.setSoftAlpha(soft);

  // Interpolate
  // solver.solve(N);

  // Copy the result back
  R = solver.getFieldPerFace();

  // Extract singularity indices
  S = solver.getSingularityIndexPerVertex();
}


IGL_INLINE void igl::copyleft::comiso::nrosy(
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
                           const bool rounding,
                           Eigen::MatrixXd& R,
                           Eigen::VectorXd& S
                           )
{
  
  // Init solver
  igl::copyleft::comiso::NRosyField solver(V,F);
  
  solver.loadk(kn);
  
  solver.setTP(TP_set);

  // Add hard constraints
  for (unsigned i=0; i<b.size();++i)
    solver.setConstraintHard(b(i),bc.row(i));

  // Interpolate
  solver.solve(p_set, p_fix, N, pj_rhs, rounding);

  // Copy the result back
  R = solver.getFieldPerFace();

  // Extract singularity indices
  S = solver.getSingularityIndexPerVertex();
  
  p_set = solver.getPeriodJump();
}

      IGL_INLINE void igl::copyleft::comiso::nrosy(
                           const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const Eigen::VectorXi& b,
                           const Eigen::MatrixXd& bc,
                           const int N,
                           Eigen::MatrixXd& R,
                           Eigen::VectorXd& S
                           ){
                             
                           }
