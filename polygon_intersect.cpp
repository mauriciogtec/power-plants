// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <cmath> 
// #include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace std;
// using namespace arma;

// [[Rcpp::export]]
inline bool in01(double t, double eps) {
  return (-eps <= t) && (t <= 1.0 + eps);
}

// [[Rcpp::export]]
inline double slope(double a1, double b1, double a2, double b2) {
  return (b2 - b1) / (a2 - a1);
}

// [[Rcpp::export]]
inline bool aeq(double a, double b, double eps) {
  return fabs(a - b) < eps;
}

// [[Rcpp::export]]
bool segment_intersect(double a1, double b1, double a2, double b2, 
                       double c1, double d1, double c2, double d2, double eps) {
  // case 1: two vertical lines
  if (a1 == a2 && c1 == c2) {
    if (!aeq(c1, a1, eps))
      return false;
    double t = (b1 - c1) / (c2 - c1);
    double s = (b2 - c1) / (c2 - c1);
    return in01(t, eps) || in01(s, eps);
  }

  // case 2: two non-vertical parallel lines
  if (a1 != a2 && c1 != c2) {
    double m1 = slope(a1, b1, a2, b2);
    double m2 = slope(c1, d1, c2, d2);
    if (m1 == m2) {
      double intercept1 = b1 - m1 * a1;
      double intercept2 = d1 - m2 * c1;
      if (!aeq(intercept1, intercept2, eps))
        return false;
      double t = (b1 - c1) / (c2 - c1);
      double s = (b2 - c1) / (c2 - c1);
      return in01(t, eps) || in01(s, eps);
    }
  }

  // case 3: two non-parallel lines
  double t2 = ((c1 - a1) * (b2 - b1) - (d1 - b1) * (a2 - a1)) / ((d2 - d1) * (a2 - a1) - (c2 - c1) * (b2 - b1)); 
  double t1 = (d1 - b1 + t2 * (d2 - d1)) / (b2 - b1);
  return in01(t1, eps) && in01(t2, eps);
}


// [[Rcpp::export]]
bool polygon_intersect(NumericMatrix coords1, NumericMatrix coords2, double eps) {
  auto n = coords1.nrow();
  auto m  = coords2.nrow();
  for (auto i = 0; i < n - 1; i ++) {
    for (auto j = 0; j < m - 1; j++) {
      double a1 = coords1(i, 0);
      double b1 = coords1(i, 1);
      double a2 = coords1(i + 1, 0);
      double b2 = coords1(i + 1, 1);
      double c1 = coords2(j, 0);
      double d1 = coords2(j, 1);
      double c2 = coords2(j + 1, 0);
      double d2 = coords2(j + 1, 1);
      if (segment_intersect(a1, b1, a2, b2, c1, d1, c2, d2, eps)) 
        return true;
    }
  }
  return false;
}


