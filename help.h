#ifndef _HELP_H_
#define _HELP_H_

#include <math.h>

struct vec2_t 
{
  bool is_transposed;
  double a_, b_;
  vec2_t( void ) : a_(0), b_(0), is_transposed(0) {}
  vec2_t( double val ) : a_(val), b_(val), is_transposed(0) {}
  vec2_t( double a, double b ) : a_(a), b_(b), is_transposed(0) {}
  double norm( void ) { return sqrt(a_ * a_ + b_ * b_);}
  double norm2( void ) { return a_ * a_ + b_ * b_;}
};

struct mat2_t 
{
  double a_, b_, c_, d_;

  mat2_t( void ) : a_(0), b_(0), c_(0), d_(0) {}
  mat2_t( double val ) : a_(val), b_(val), c_(val), d_(val) {}
  mat2_t( double a, double b, double c, double d ) : a_(a), b_(b), c_(c), d_(d) {}
  mat2_t inverse( void )
  {
    double const del = a_ * d_ - b_ * c_;
    return mat2_t(d_ / del, -b_ / del, -c_ / del, a_ / del);
  }
};

vec2_t operator*(vec2_t &v, double num )
{
  return vec2_t(v.a_ * num, v.b_ * num);
}
vec2_t operator*(double num, vec2_t &v )
{
  return vec2_t(v.a_ * num, v.b_ * num);
}
vec2_t operator-( vec2_t &v1, vec2_t &v2 )
{
  return vec2_t(v1.a_ - v2.a_, v1.b_ - v2.b_);
}
vec2_t operator+( vec2_t &v1, vec2_t &v2 )
{
  return vec2_t(v1.a_ + v2.a_, v1.b_ + v2.b_);
}

vec2_t operator*(mat2_t &m, vec2_t &v )
{
  return vec2_t(m.a_ * v.a_ + m.b_ * v.b_, m.c_ * v.a_ + m.d_ * v.b_);
}
vec2_t operator*( vec2_t &v, mat2_t &m )
{
  vec2_t res(m.a_ * v.a_ + m.c_ * v.b_, m.b_ * v.a_ + m.d_ * v.b_);
  res.is_transposed = true;
  return res;
}

#endif /* _HELP_H_ */
