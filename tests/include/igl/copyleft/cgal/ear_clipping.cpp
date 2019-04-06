#include <test_common.h>
#include <igl/copyleft/cgal/ear_clipping.h>

TEST_CASE("ear_clipping: boolean", "[igl/copyleft/cgal]")
{
  // Example1: simple polygon 
  Eigen::MatrixXd polygon(10,2);
  polygon<<2,-3,4,1,5.5,-2,6,2.5,5,1,4,5,3,0,1,1,1,5,0,0;
  Eigen::VectorXi RT,nR,M;
  Eigen::MatrixXi eF;
  Eigen::MatrixXd nP;
  RT.setZero(polygon.rows());
  igl::copyleft::cgal::ear_clipping(polygon,RT,M,eF,nP);
  REQUIRE(nP.rows() == 0);

  // Example 2: polygon with colinear edges
  

}
