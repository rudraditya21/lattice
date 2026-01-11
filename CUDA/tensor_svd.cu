#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_svd(const scalar_t* a, scalar_t* u, scalar_t* s, scalar_t* v,
                                       lattice_svd_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (params.m != 2 || params.n != 2) return;
  scalar_t a0 = a[0];
  scalar_t b0 = a[1];
  scalar_t c0 = a[2];
  scalar_t d0 = a[3];
  scalar_t s1s1 = a0 * a0 + c0 * c0;
  scalar_t s2s2 = b0 * b0 + d0 * d0;
  scalar_t off = a0 * b0 + c0 * d0;
  scalar_t tr = s1s1 + s2s2;
  scalar_t det = s1s1 * s2s2 - off * off;
  scalar_t disc = tr * tr * static_cast<scalar_t>(0.25) - det;
  if (disc < static_cast<scalar_t>(0)) disc = static_cast<scalar_t>(0);
  scalar_t tmp = lattice_sqrt(disc);
  scalar_t s1 = lattice_sqrt(tr * static_cast<scalar_t>(0.5) + tmp);
  scalar_t s2 = lattice_sqrt(tr * static_cast<scalar_t>(0.5) - tmp);
  s[0] = s1;
  s[1] = static_cast<scalar_t>(0);
  s[2] = static_cast<scalar_t>(0);
  s[3] = s2;

  scalar_t ata00 = s1s1;
  scalar_t ata01 = off;
  scalar_t ata11 = s2s2;
  if (lattice_abs(ata01) < static_cast<scalar_t>(1e-6)) {
    v[0] = static_cast<scalar_t>(1);
    v[1] = static_cast<scalar_t>(0);
    v[2] = static_cast<scalar_t>(0);
    v[3] = static_cast<scalar_t>(1);
  } else {
    scalar_t t = (ata00 - ata11) / (static_cast<scalar_t>(2) * ata01);
    scalar_t sign =
        t >= static_cast<scalar_t>(0) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1);
    scalar_t tau = sign / (lattice_abs(t) + lattice_sqrt(static_cast<scalar_t>(1) + t * t));
    scalar_t cs = static_cast<scalar_t>(1) / lattice_sqrt(static_cast<scalar_t>(1) + tau * tau);
    scalar_t sn = cs * tau;
    v[0] = cs;
    v[1] = -sn;
    v[2] = sn;
    v[3] = cs;
  }

  scalar_t v1x = v[0];
  scalar_t v1y = v[2];
  scalar_t v2x = v[1];
  scalar_t v2y = v[3];
  scalar_t u1x = a0 * v1x + b0 * v1y;
  scalar_t u1y = c0 * v1x + d0 * v1y;
  scalar_t u2x = a0 * v2x + b0 * v2y;
  scalar_t u2y = c0 * v2x + d0 * v2y;
  scalar_t n1 = lattice_sqrt(u1x * u1x + u1y * u1y);
  scalar_t n2 = lattice_sqrt(u2x * u2x + u2y * u2y);
  if (n1 > static_cast<scalar_t>(0)) {
    u[0] = u1x / n1;
    u[2] = u1y / n1;
  } else {
    u[0] = static_cast<scalar_t>(0);
    u[2] = static_cast<scalar_t>(0);
  }
  if (n2 > static_cast<scalar_t>(0)) {
    u[1] = u2x / n2;
    u[3] = u2y / n2;
  } else {
    u[1] = static_cast<scalar_t>(0);
    u[3] = static_cast<scalar_t>(0);
  }
}
