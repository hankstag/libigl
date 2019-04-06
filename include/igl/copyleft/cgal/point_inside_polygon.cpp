// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2019 Hanxiao Shen <hanxiao@cs.nyu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "point_inside_polygon.h"

// Ray casting algorithm
// [https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/]
template <typename Scalar>
IGL_INLINE bool igl::copyleft::cgal::point_inside_polygon(
    const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& P,
    const Eigen::Matrix<Scalar,1,2>& q
){
    // There must be at least 3 vertices in polygon[]
    if (P.rows() < 3)  return false;
    
    // pick a far right vertex (outside P)
    Scalar r = P.col(0).maxCoeff() + 2.0f;
    Eigen::Matrix<Scalar,1,2> q2(2);

    q2 << r, q(1);
    int count = 0;
    for(int i=0;i<P.rows();i++){
        Eigen::Matrix<Scalar,1,2> a = P.row(i);
        Eigen::Matrix<Scalar,1,2> b = P.row((i+1)%P.rows());
        if(segment_segment_intersect(a,b,q,q2,0)){
            count++;
        }
    }
    return count&1;
}

#ifdef IGL_STATIC_LIBRARY
template bool igl::copyleft::cgal::point_inside_polygon<double>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, 1, 2, 1, 1, 2> const&);
#endif