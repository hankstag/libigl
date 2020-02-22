// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "dijkstra.h"

template <typename IndexType, typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra_with_len(
  const IndexType &source,
  const std::set<IndexType> &targets,
  const Eigen::SparseMatrix<double>& G,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous)
{
  int numV = G.size();
  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    for (Eigen::SparseMatrix<double>::InnerIterator it(G,u); it; ++it){
      IndexType v = it.row();
      // it.row();   // row index
      // it.col();   // col index (here it is equal to k)
      typename DerivedD::Scalar distance_through_u = dist + it.value();
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  //we should never get here
  return -1;
}

template <typename IndexType, typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra(
  const IndexType &source,
  const std::set<IndexType> &targets,
  const std::vector<std::vector<IndexType> >& VV,
  const std::vector<double>& weights,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous)
{
  int numV = VV.size();
  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      typename DerivedD::Scalar distance_through_u = dist + weights[u];
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  //we should never get here
  return -1;
}

template <typename IndexType, typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra(
  const IndexType &source,
  const std::set<IndexType> &targets,
  const std::vector<std::vector<IndexType> >& VV,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous)
{
  std::vector<double> weights(VV.size(), 1.0);
  return dijkstra(source, targets, VV, weights, min_distance, previous);
}

template <typename IndexType, typename DerivedP>
IGL_INLINE void igl::dijkstra(
  const IndexType &vertex,
  const Eigen::MatrixBase<DerivedP> &previous,
  std::vector<IndexType> &path)
{
  IndexType source = vertex;
  path.clear();
  for ( ; source != -1; source = previous[source])
    path.push_back(source);
}

template <typename IndexType, typename DerivedV,
typename DerivedD, typename DerivedP, typename DerivedW>
IGL_INLINE int igl::dijkstra(
  const Eigen::MatrixBase<DerivedV> &V,
  std::map<std::pair<IndexType,IndexType>,double>& metric,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous,
  bool use_uv_metric,
  const Eigen::MatrixBase<DerivedW> &W
){
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      double e_len = 0;
      if(!use_uv_metric)
        e_len = (V.row(u) - V.row(v)).norm();
      else{
        int a = u, b = v;
        assert(metric.find(std::make_pair(a,b)) != metric.end());
        e_len = metric[std::make_pair(a,b)];
      }
      typename DerivedD::Scalar distance_through_u = dist + e_len;
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

// dijkstra on M2 - preventing from self-intersection
template <typename IndexType, typename DerivedV,
typename DerivedD, typename DerivedP, typename DerivedW>
IGL_INLINE int igl::dijkstra_m2(
  const Eigen::MatrixBase<DerivedV> &V,
  std::map<std::pair<IndexType,IndexType>,double>& metric,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous,
  bool use_uv_metric,
  const Eigen::MatrixBase<DerivedW> &W
){
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      
      // // check intersection
      bool intersect = false;
      int p = u;
      while(p != -1){
        p = previous[p];
        if(p/2 == v/2){
          intersect = true;
        }
      }
      if(intersect) continue;
      
      double e_len = 0;
      if(!use_uv_metric)
        e_len = (V.row(u) - V.row(v)).norm();
      else{
        int a = u, b = v;
        assert(metric.find(std::make_pair(a,b)) != metric.end());
        e_len = metric[std::make_pair(a,b)];
      }
      typename DerivedD::Scalar distance_through_u = dist + e_len;
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

// dijkstra on M2 - preventing from self-intersection
template <typename IndexType, typename DerivedV,
typename DerivedD, typename DerivedP, typename DerivedW>
IGL_INLINE int igl::dijkstra_m2(
  const Eigen::MatrixBase<DerivedV> &V,
  std::map<std::pair<IndexType,IndexType>,double>& metric,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous,
  bool use_uv_metric,
  const Eigen::MatrixBase<DerivedW> &W,
  bool need_cut, 
  const Eigen::VectorXi& J, 
  const Eigen::VectorXi& cut_map
){
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      
      // check intersection
      bool intersect = false;
      int p = u;
      while(p != -1){
        p = previous[p];
        if(p == -1) break;
        int v0 = need_cut ? J(cut_map(p)) : J(p);
        int v1 = need_cut ? J(cut_map(v)) : J(v);
        if(v0/2 == v1/2 && targets.find(v) == targets.end()){
          intersect = true;
        }
      }
      if(intersect) continue;
      
      double e_len = 0;
      if(!use_uv_metric)
        e_len = (V.row(u) - V.row(v)).norm();
      else{
        int a = u, b = v;
        assert(metric.find(std::make_pair(a,b)) != metric.end());
        e_len = metric[std::make_pair(a,b)];
      }
      typename DerivedD::Scalar distance_through_u = dist + e_len;
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

template <typename IndexType, typename DerivedV,
typename DerivedD, typename DerivedP, typename DerivedW>
IGL_INLINE int igl::dijkstra(
  const Eigen::MatrixBase<DerivedV> &V,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  const Eigen::MatrixBase<DerivedW> &W,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous)
{
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      double e_len = (V.row(u) - V.row(v)).norm();
      typename DerivedD::Scalar distance_through_u = dist + ((W(v) + W(u)) / 2)*e_len;
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}


template <typename IndexType, typename DerivedV,
typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra(
  const Eigen::MatrixBase<DerivedV> &V,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous)
{
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      typename DerivedD::Scalar distance_through_u = dist + (V.row(u) - V.row(v)).norm();
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

template <typename IndexType, typename DerivedV, typename DerivedI,
typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra_tree(
  const Eigen::MatrixBase<DerivedV> &V,
  const std::vector<std::vector<IndexType> >& VV,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous,
  const Eigen::MatrixBase<DerivedI> &is_leaf)
{
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      typename DerivedD::Scalar distance_through_u = dist + (V.row(u) - V.row(v)).norm();
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        if(!is_leaf(v))
          vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

template <typename IndexType, typename DerivedV, typename DerivedI,
typename DerivedD, typename DerivedP>
IGL_INLINE int igl::dijkstra_tree(
  const Eigen::MatrixBase<DerivedV> &V,
  const std::vector<std::vector<IndexType> >& VV,
  std::map<std::pair<int,int>, double>& metric,
  const IndexType &source,
  const std::set<IndexType> &targets,
  Eigen::PlainObjectBase<DerivedD> &min_distance,
  Eigen::PlainObjectBase<DerivedP> &previous,
  const Eigen::MatrixBase<DerivedI> &is_leaf)
{
  int numV = VV.size();

  min_distance.setConstant(numV, 1, std::numeric_limits<typename DerivedD::Scalar>::infinity());
  min_distance[source] = 0;
  previous.setConstant(numV, 1, -1);
  std::set<std::pair<typename DerivedD::Scalar, IndexType> > vertex_queue;
  vertex_queue.insert(std::make_pair(min_distance[source], source));

  while (!vertex_queue.empty())
  {
    typename DerivedD::Scalar dist = vertex_queue.begin()->first;
    IndexType u = vertex_queue.begin()->second;
    vertex_queue.erase(vertex_queue.begin());

    if (targets.find(u)!= targets.end())
      return u;

    // Visit each edge exiting u
    const std::vector<int> &neighbors = VV[u];
    for (std::vector<int>::const_iterator neighbor_iter = neighbors.begin();
         neighbor_iter != neighbors.end();
         neighbor_iter++)
    {
      IndexType v = *neighbor_iter;
      typename DerivedD::Scalar distance_through_u = dist + metric[std::make_pair(u,v)];// (V.row(u) - V.row(v)).norm();
      if (distance_through_u < min_distance[v]) {
        vertex_queue.erase(std::make_pair(min_distance[v], v));

        min_distance[v] = distance_through_u;
        previous[v] = u;
        if(!is_leaf(v))
          vertex_queue.insert(std::make_pair(min_distance[v], v));

      }

    }
  }
  return -1;
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
template int igl::dijkstra<int, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(int const&, std::set<int, std::less<int>, std::allocator<int> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template void igl::dijkstra<int, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(int const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, std::vector<int, std::allocator<int> >&);
template int igl::dijkstra<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template int igl::dijkstra<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
// template int igl::dijkstra<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template int igl::dijkstra_tree<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template int igl::dijkstra_with_len<int, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::SparseMatrix<double, 0, int> const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template int igl::dijkstra<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::map<std::pair<int, int>, double, std::less<std::pair<int, int> >, std::allocator<std::pair<std::pair<int, int> const, double> > >&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, bool, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template int igl::dijkstra_m2<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::map<std::pair<int, int>, double, std::less<std::pair<int, int> >, std::allocator<std::pair<std::pair<int, int> const, double> > >&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, bool, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
template int igl::dijkstra_m2<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::map<std::pair<int, int>, double, std::less<std::pair<int, int> >, std::allocator<std::pair<std::pair<int, int> const, double> > >&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, bool, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, bool, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&);
template int igl::dijkstra_tree<int, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&, std::map<std::pair<int, int>, double, std::less<std::pair<int, int> >, std::allocator<std::pair<std::pair<int, int> const, double> > >&, int const&, std::set<int, std::less<int>, std::allocator<int> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&);
#endif
