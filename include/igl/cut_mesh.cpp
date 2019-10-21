// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2019 Hanxiao Shen <hanxiao@cims.nyu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include <igl/cut_mesh.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/HalfEdgeIterator.h>

// wrapper for input/output style
template <typename DerivedV, typename DerivedF, typename DerivedC>
IGL_INLINE void igl::cut_mesh(
  const Eigen::MatrixBase<DerivedV>& V,
  const Eigen::MatrixBase<DerivedF>& F,
  const Eigen::MatrixBase<DerivedC>& C,
  Eigen::PlainObjectBase<DerivedV>& Vn,
  Eigen::PlainObjectBase<DerivedF>& Fn
){
  Vn = V;
  Fn = F;
  cut_mesh(Vn,Fn,C);
}


// cut mesh - in place update
template <typename DerivedV, typename DerivedF, typename DerivedC>
IGL_INLINE void igl::cut_mesh(
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedF>& F,
  const Eigen::MatrixBase<DerivedC>& C
){
  
  typedef typename DerivedF::Scalar Index;
  DerivedF FF, FFi;
  
  igl::triangle_triangle_adjacency(F,FF,FFi);
  
  // target number of occurance of each vertex
  Eigen::Matrix<Index,Eigen::Dynamic,1> g(V.rows());
  g.setZero();
  
  // current number of occurance of each vertex as the alg proceed
  Eigen::Matrix<Index,Eigen::Dynamic,1> o(V.rows());
  o.setConstant(1);
  
  // initialize g
  for(Index i=0;i<F.rows();i++){
    for(Index k=0;k<3;k++){
      if(C(i,k) == 1){
        Index u = F(i,k);
        Index v = F(i,(k+1)%3);
        if(u > v) continue; // only compute every (undirected) edge ones
        g(u) += 1;
        g(v) += 1;
      }
    }
  }
  
  Index n_v = V.rows(); // original number of vertices
  
  // estimate number of new vertices 
  // and resize V
  Index n_new = 0;
  for(Index i=0;i<g.rows();i++)
    n_new += ((g(i) > 0) ? g(i)-1 : 0);
  V.conservativeResize(n_v+n_new,Eigen::NoChange);
  
  // pointing to the current bottom of V
  Index pos = n_v;
  for(Index f=0;f<C.rows();f++){
    for(Index k=0;k<3;k++){
      Index v0 = F(f,k);
      if(F(f,k) >= n_v) continue; // ignore new vertices
      if(C(f,k) == 1 && o(v0) != g(v0)){
        igl::HalfEdgeIterator<DerivedF,DerivedF,DerivedF> he(F,FF,FFi,f,k);
                              
        // rotate clock-wise around v0 until hit another cut
        std::vector<Index> fan;
        Index fi = he.Fi();
        Index ei = he.Ei();
        do{
          fan.push_back(fi);
          he.flipE();
          he.flipF();
          fi = he.Fi();
          ei = he.Ei();
        }while(C(fi,ei) == 0);
        
        // if cuts form a sector/fan
        if(fi != f){
          // make a copy
          // V.conservativeResize(V.rows()+1,V.cols());
          V.row(pos) << V.row(v0);
          // V.row(V.rows()-1) << V.row(v0);
          
          // add one occurance to v0
          o(v0) += 1;
          
          // replace old v0
          for(Index f0: fan)
            for(Index j=0;j<3;j++)
              if(F(f0,j) == v0)
                F(f0,j) = pos;
          
          // mark cuts as boundary
          FF(f,k) = -1;
          FF(fi,ei) = -1;
          
          pos++;
        }
      }
    }
  }
  
}


#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template void igl::cut_mesh<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&);
template void igl::cut_mesh<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> >&);
template void igl::cut_mesh<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template void igl::cut_mesh<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
#endif
