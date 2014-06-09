/* Sergeev Artemiy */

#include <stdio.h>
#include "help.h"

/* (3 - sqrt(5)) / 2 constant value */
#define PHI2 0.38196601125010515179541316563436
#define EPS 1e-16

unsigned int iter;

struct my_func
{
  double a, b, c, d;

  my_func( double a_, double b_, double c_, double d_ ) : a(a_), b(b_), c(c_), d(d_)
  {
  }
  double operator() ( vec2_t &x )
  {
    return a * x.a_ * x.a_ + b * x.b_ * x.b_ + exp(c * x.a_ + d * x.b_);
  }
  vec2_t grad( vec2_t &x )
  {
    return vec2_t(2 * a * x.a_ + c * exp(c * x.a_ + d * x.b_), 2 * b * x.b_ + d * exp(c * x.a_ + d * x.b_));
  }
  mat2_t hessian( vec2_t &x )
  {
    double const val1 = 2 * a + c * c * exp(c * x.a_ + d * x.b_),
                 val2 =         c * d * exp(c * x.a_ + d * x.b_),
                 val3 = 2 * b + d * d * exp(c * x.a_ + d * x.b_);
    return mat2_t(val1, val2, val2, val3);
  }
};

bool ls( double v1, double v2 )
{
  return v1 - v2 < EPS;
}

vec2_t grad_frag_step( my_func f, vec2_t start, double lambda, double epsilon, double precision )
{
  vec2_t point = start;
  vec2_t x(-0.313, 0.156);

  iter = 0;

  double val = f.grad(point).norm();

  for(;;)
  {
    vec2_t grad_val = f.grad(point);
    double grad_norm = grad_val.norm(), step = 1;

    printf("%lf\n", val / (point - start).norm());

    if (grad_norm < precision)
      break;

    vec2_t next_point;
    for (;;)
    {
      next_point = point - step * grad_val;
      double diff = f(next_point) - f(point),
             bound = -epsilon * step * grad_norm * grad_norm;
      if (ls(diff, bound))
        break;
      step *= lambda;
    }
    point = next_point;
    ++iter;
  }
  return point;
}

mat2_t get_next_A( my_func f, mat2_t A, vec2_t point, vec2_t next_point )
{
  vec2_t v = next_point - point, u = f.grad(next_point) - f.grad(point);

  double del1 = (v.a_ * u.a_ + v.b_ * u.b_);
  mat2_t first(v.a_ * v.a_ / del1, v.a_ * v.b_ / del1, v.a_ * v.b_ / del1, v.b_ * v.b_ / del1);
  double del2 = A.a_ * u.a_ * u.a_ + 
                (A.b_ + A.c_) * u.a_ * u.b_ + A.d_ * u.b_ * u.b_;
  double a = A.a_ * u.a_ + A.b_ * u.b_,
         b = A.c_ * u.a_ + A.d_ * u.b_,
         c = A.a_ * u.a_ + A.c_ * u.b_,
         d = A.b_ * u.a_ + A.d_ * u.b_;
  mat2_t second(a * c / del2, a * d / del2, b * c / del2, b * d / del2);

  A.a_ = A.a_ + first.a_ - second.a_;
  A.b_ = A.b_ + first.b_ - second.b_;
  A.c_ = A.c_ + first.c_ - second.c_;
  A.d_ = A.d_ + first.d_ - second.d_;
  return A;
}

vec2_t grad_frag_step_mod( my_func f, vec2_t start, double lambda, double epsilon, double precision )
{
  vec2_t point = start;
  mat2_t A(1, 0, 0, 1);

  iter = 0;

  for(;;)
  {
    vec2_t grad_val = f.grad(point);
    double grad_norm = grad_val.norm(), step = 1;

    if (grad_norm < precision)
      break;

    vec2_t temp = A * grad_val;
    double temp_norm = temp.norm();
    vec2_t next_point;
    for (;;)
    {
      next_point = point - step * temp;
      double diff = f(next_point) - f(point),
             bound = -epsilon * step * temp_norm * temp_norm;
      if (diff < bound)
        break;
      step *= lambda;
    }
    A = get_next_A(f, A, point, next_point);
    point = next_point;

    ++iter;
  }
  return point;
}

template <typename functor_t>
double golden_section( functor_t f, double start_a, double start_b, double precision )
{
  double y = start_a + PHI2 * (start_b - start_a),
         z = start_b - PHI2 * (start_b - start_a), 
         fy = f(y), fz = f(z);
  double a = start_a, b = start_b;

  while (b - a >= precision)
  {
    if (fy <= fz)
    {
      b = z;
      z = y;
      fz = fy;
      y = a + PHI2 * (b - a);
      fy = f(y);
    }
    else
    {
      a = y;
      y = z;
      fy = fz;
      z = b - PHI2 * (b - a);
      fz = f(z);
    }
  }
  return (a + b) / 2;
}

vec2_t grad_descent( my_func f, vec2_t start, double precision )
{
  vec2_t point = start;

  iter = 0;

  for(;;)
  {
    vec2_t grad_val = f.grad(point);
    double grad_norm = grad_val.norm2();
    mat2_t hessian_inv = f.hessian(point).inverse();

    if (grad_norm < precision)
      break;

    vec2_t dir = hessian_inv * grad_val * (-1);
    struct auxiliary_problem
    {
      vec2_t &dir, &point;
      my_func &f;
      auxiliary_problem( vec2_t &dir_, vec2_t &point_, my_func &f_ ) : dir(dir_), point(point_), f(f_) {}
      double operator()( double step_ )
      {
        return f(point + step_ * dir);
      }
    };
    double step = golden_section(auxiliary_problem(dir, point, f), 0, 1, precision);
    point = point + step * dir;
    ++iter;
  }
  return point;
}

int main( void )
{
  my_func f(1, 2, 1, -1);
  vec2_t res;

  
  /*res = grad_frag_step(f, vec2_t(5, 5), 0.5, 0.5, 0.1);
  printf("%.4lf %6.2lf %6.2lf %6.2lf %u\n", 0.1, res.a_, res.b_, f(res), iter);*/
  res = grad_frag_step(f, vec2_t(-10, 10), 0.5, 0.5, 0.01);
  /*printf("%.4lf %6.3lf %6.3lf %6.3lf %u\n", 0.01, res.a_, res.b_, f(res), iter);
  res = grad_frag_step(f, vec2_t(5, 5), 0.5, 0.5, 0.001);
  printf("%.4lf %6.4lf %6.4lf %6.4lf %u\n", 0.001, res.a_, res.b_, f(res), iter);
  res = grad_frag_step(f, vec2_t(5, 5), 0.5, 0.5, 0.0000000001);
  printf("%.10lf %6.4lf %6.4lf %6.4lf %u\n", 0.0000000001, res.a_, res.b_, f(res), iter);

  printf("-------------\n");

  res = grad_descent(f, vec2_t(5, 5), 0.1);
  printf("%.4lf %6.2lf %6.2lf %6.2lf %u\n", 0.1, res.a_, res.b_, f(res), iter);
  res = grad_descent(f, vec2_t(5, 5), 0.01);
  printf("%.4lf %6.3lf %6.3lf %6.3lf %u\n", 0.01, res.a_, res.b_, f(res), iter);
  res = grad_descent(f, vec2_t(5, 5), 0.001);
  printf("%.4lf %6.4lf %6.4lf %6.4lf %u\n", 0.001, res.a_, res.b_, f(res), iter);
  res = grad_descent(f, vec2_t(5, 5), 0.0000000001);
  printf("%.10lf %6.4lf %6.4lf %6.4lf %u\n", 0.0000000001, res.a_, res.b_, f(res), iter);

  printf("-------------\n");

  res = grad_frag_step_mod(f, vec2_t(5, 5), 0.5, 0.5, 0.1);
  printf("%.4lf %6.2lf %6.2lf %6.2lf %u\n", 0.1, res.a_, res.b_, f(res), iter);
  res = grad_frag_step_mod(f, vec2_t(5, 5), 0.5, 0.5, 0.01);
  printf("%.4lf %6.3lf %6.3lf %6.3lf %u\n", 0.01, res.a_, res.b_, f(res), iter);
  res = grad_frag_step_mod(f, vec2_t(5, 5), 0.5, 0.5, 0.001);
  printf("%.4lf %6.4lf %6.4lf %6.4lf %u\n", 0.001, res.a_, res.b_, f(res), iter);
  res = grad_frag_step_mod(f, vec2_t(5, 5), 0.5, 0.5, 0.0000000001);
  printf("%.10lf %6.4lf %6.4lf %6.4lf %u\n", 0.0000000001, res.a_, res.b_, f(res), iter);*/

  return 0;
}