import sys
import os
import typing

import numpy
import scipy
import xarray as xr

#%%% S. Mae 2025.3.9
#from spline import Spline

class Spline:
    def __init__(self, ns_b: int):
        self.x1_sp = numpy.zeros(ns_b)
        self.y1_sp = numpy.zeros(ns_b - 1)
        self.a_sp = numpy.zeros(ns_b - 1)
        self.b_sp = numpy.zeros(ns_b - 1)
        self.c_sp = numpy.zeros(ns_b)

        self.ah = numpy.zeros(ns_b)
        self.bh = numpy.zeros(ns_b)
        self.ch = numpy.zeros(ns_b)
        self.dh = numpy.zeros(ns_b)
        self.G = numpy.zeros(ns_b)
        self.H = numpy.zeros(ns_b)

        self.nmax = ns_b
        self.h1_h = numpy.zeros(ns_b - 2)
        self.h2_h = numpy.zeros(ns_b - 2)
        self.h1_sp = numpy.zeros(ns_b - 1)
        self.h2_sp = numpy.zeros(ns_b - 1)

    def cubic_spline_pre(
            self,
            xp: numpy.ndarray,
            yp: numpy.ndarray,
            n: int):
        h1 = xp[1] - xp[0]
        self.bh[0] = 2.0 * h1
        self.ch[0] = h1
        self.dh[0] = 3.0 * (yp[1] - yp[0])

        self.h1_h[:] = xp[1:-1] - xp[:-2]
        self.h2_h[:] = xp[2:] - xp[1:-1]
        self.ah[1:-1] = self.h2_h
        self.bh[1:-1] = 2.0 * (xp[2:] - xp[:-2])
        self.ch[1:-1] = self.h1_h
        self.dh[1:-1] = 3.0 * ((yp[1:-1] - yp[0:-2]) * self.h2_h / self.h1_h +
                               (yp[2:] - yp[1:-1]) * self.h1_h / self.h2_h)

        h1 = xp[n - 1] - xp[n - 2]
        self.ah[n - 1] = h1
        self.bh[n - 1] = 2.0 * h1
        self.dh[n - 1] = 3.0 * (yp[n - 1] - yp[n - 2])

        self.tridiagonal_matrix(self.ah, self.bh, self.ch, self.dh, n)

        self.h1_sp[:] = xp[1:] - xp[:-1]
        self.h2_sp[:] = self.h1_sp * self.h1_sp
        self.x1_sp[:-1] = xp[:-1]
        self.y1_sp = yp[:-1]
        self.b_sp = (3.0 * (yp[1:] - yp[:-1]) -
                     (self.c_sp[1:] + 2.0 * self.c_sp[:-1]) * self.h1_sp) / self.h2_sp
        self.a_sp = (self.c_sp[1:] - self.c_sp[:-1] -
                     2.0 * self.b_sp * self.h1_sp) / (3.0 * self.h2_sp)
        self.x1_sp[n - 1] = xp[n - 1]

        self.nmax = n

    def cubic_spline(self, x: float, y: float, dydx: float) -> tuple[float, float]:
        i1: int = None
        i2: int = None
        if x <= self.x1_sp[1]:
            i1 = 0
            if x < self.x1_sp[0]:
                print("Out of bounds -- cubic_spline", x)
        elif x >= self.x1_sp[self.nmax - 2]:
            i1 = self.nmax - 2
            if x > self.x1_sp[self.nmax - 1]:
                print("Out of bounds -- cubic_spline", x)
        else:
            i1 = 1
            i2 = self.nmax - 2
            while i2 - i1 > 1:
                i = (i1 + i2) // 2
                if x < self.x1_sp[i]:
                    i2 = i
                else:
                    i1 = i

        x0 = x - self.x1_sp[i1]
        y = ((self.a_sp[i1] * x0 + self.b_sp[i1]) *
             x0 + self.c_sp[i1]) * x0 + self.y1_sp[i1]
        dydx = (3.0 * self.a_sp[i1] * x0 + 2.0 *
                self.b_sp[i1]) * x0 + self.c_sp[i1]

        return y, dydx

    def cubic_spline_all(
            self,
            nrho: int,
            x: numpy.ndarray,
            y: numpy.ndarray,
            dydx: numpy.ndarray):
        for js in range(nrho):
            y[js], dydx[js] = self.cubic_spline(x[js], y[js], dydx[js])

    def tridiagonal_matrix(
            self,
            a1: numpy.ndarray,
            b1: numpy.ndarray,
            c1: numpy.ndarray,
            d1: numpy.ndarray,
            n: int):
        self.G[0] = - c1[0] / b1[0]
        self.H[0] = d1[0] / b1[0]

        for i in range(1, n):
            den = 1.0 / (b1[i] + a1[i] * self.G[i - 1])
            self.G[i] = - c1[i] * den
            self.H[i] = (d1[i] - a1[i] * self.H[i - 1]) * den

        self.c_sp[n - 1] = self.H[n - 1]
        for i in range(n - 2, -1, -1):
            self.c_sp[i] = self.G[i] * self.c_sp[i + 1] + self.H[i]
#%%%


def format_double(x):
    e = numpy.format_float_scientific(
        x, precision=15, unique=True, exp_digits=3, min_digits=15)
    if x >= 0:
        return f' {e}'.upper()
    else:
        return e.upper()


class Boozmn:
    def __init__(self):
        self.nfp_b: int = 0
        self.ns_b: int = 0
        self.aspect_b: float = 0.0
        self.rmax_b: float = 0.0
        self.rmin_b: float = 0.0
        self.betaxis_b: float = 0.0

        self.iota_b_nu: numpy.ndarray = None
        self.pres_b_nu: numpy.ndarray = None
        self.beta_b_nu: numpy.ndarray = None
        self.phip_b_nu: numpy.ndarray = None
        self.phi_b_nu: numpy.ndarray = None
        self.bvco_b_nu: numpy.ndarray = None
        self.buco_b_nu: numpy.ndarray = None

        self.mboz_b: int = 0
        self.nboz_b: int = 0
        self.mnboz_b: int = 0
        self.jsize: int = 0

        self.version: str = ''
        self.lasym_b: bool = False

        self.bmnc_b: numpy.ndarray = None
        self.rmnc_b: numpy.ndarray = None
        self.zmns_b: numpy.ndarray = None
        self.pmns_b: numpy.ndarray = None
        self.gmnc_b: numpy.ndarray = None
        self.ixm_b: numpy.ndarray = None
        self.ixn_b: numpy.ndarray = None
        self.jlist: numpy.ndarray = None

        # lasym_b
        self.bmns_b: numpy.ndarray = None
        self.rmns_b: numpy.ndarray = None
        self.zmnc_b: numpy.ndarray = None
        self.pmnc_b: numpy.ndarray = None
        self.gmns_b: numpy.ndarray = None


# --- for metric calc.
class Metric:
    def __init__(self, olog: typing.TextIO):
        # for extrapolation
        self.bbozc_nu: numpy.ndarray = None
        self.rbozc_nu: numpy.ndarray = None
        self.zbozs_nu: numpy.ndarray = None
        self.pbozs_nu: numpy.ndarray = None
        self.gbozc_nu: numpy.ndarray = None

        self.bbozs_nu: numpy.ndarray = None
        self.rbozs_nu: numpy.ndarray = None
        self.zbozc_nu: numpy.ndarray = None
        self.pbozc_nu: numpy.ndarray = None
        self.gbozs_nu: numpy.ndarray = None

        # rho theta_b zeta_b grids and q-profile
        self.rho: numpy.ndarray = None
        self.rho_nu: numpy.ndarray = None
        self.rho2: numpy.ndarray = None
        self.qq_nu: numpy.ndarray = None

        # --- for interpolation
        self.qq: numpy.ndarray = None
        self.dqdrho: numpy.ndarray = None
        self.shat: numpy.ndarray = None
        self.epst: numpy.ndarray = None

        self.cug: numpy.ndarray = None
        self.cui: numpy.ndarray = None
        self.dummy1: numpy.ndarray = None
        self.phi_b: numpy.ndarray = None
        self.dphidrho: numpy.ndarray = None

        self.bbozc: numpy.ndarray = None
        self.rbozc: numpy.ndarray = None
        self.zbozs: numpy.ndarray = None
        self.pbozs: numpy.ndarray = None
        self.gbozc: numpy.ndarray = None

        self.dbbozc: numpy.ndarray = None
        self.drbozc: numpy.ndarray = None
        self.dzbozs: numpy.ndarray = None
        self.dpbozs: numpy.ndarray = None
        self.dummy2: numpy.ndarray = None

        # if lasym_b
        self.bbozs: numpy.ndarray = None
        self.rbozs: numpy.ndarray = None
        self.zbozc: numpy.ndarray = None
        self.pbozc: numpy.ndarray = None
        self.gbozs: numpy.ndarray = None

        self.dbbozs: numpy.ndarray = None
        self.drbozs: numpy.ndarray = None
        self.dzbozc: numpy.ndarray = None
        self.dpbozc: numpy.ndarray = None
        # ---

        # --- grids and 3D data
        # N.B.: theta, zeta in f90 are 0-indexed
        self.theta: numpy.ndarray = None
        self.zeta: numpy.ndarray = None

        self.Bax: float = 0.0
        self.Rax: float = 0.0
        self.aa: float = 0.0

        # --- B(rho,theta_b,zeta_b), R(rho,theta_b,zeta_b), Z(rho,theta_b,zeta_b), phi(rho,theta_b,zeta_b)
        self.bb: numpy.ndarray = None
        self.rr: numpy.ndarray = None
        self.zz: numpy.ndarray = None
        self.ph: numpy.ndarray = None
        self.ggb: numpy.ndarray = None

        self.dbb_drho: numpy.ndarray = None
        self.drr_drho: numpy.ndarray = None
        self.dzz_drho: numpy.ndarray = None
        self.dph_drho: numpy.ndarray = None

        self.dbb_dtht: numpy.ndarray = None
        self.drr_dtht: numpy.ndarray = None
        self.dzz_dtht: numpy.ndarray = None
        self.dph_dtht: numpy.ndarray = None

        self.dbb_dzeta: numpy.ndarray = None
        self.drr_dzeta: numpy.ndarray = None
        self.dzz_dzeta: numpy.ndarray = None
        self.dph_dzeta: numpy.ndarray = None

        # --- metric_boozer
        self.ggup_boz: numpy.ndarray = None
        self.ggdn_boz: numpy.ndarray = None
        self.ggsq_boz: numpy.ndarray = None
        self.rootg_boz: numpy.ndarray = None
        self.rootg_boz0: numpy.ndarray = None  # another definition
        self.rootg_boz1: numpy.ndarray = None

        self.olog = olog

    def extrapolation_to_magnetic(self, boozmn: Boozmn):
        # --- extrapolation to the magnetic axis
        self.bbozc_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
        self.rbozc_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
        self.zbozs_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
        self.pbozs_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
        self.gbozc_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))

        if boozmn.lasym_b:
            self.bbozs_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
            self.rbozs_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
            self.zbozc_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
            self.pbozc_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))
            self.gbozs_nu = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b))

        # --- copy array
        self.bbozc_nu[:, 1:] = boozmn.bmnc_b  # Non-Uniform grid
        self.rbozc_nu[:, 1:] = boozmn.rmnc_b
        self.zbozs_nu[:, 1:] = boozmn.zmns_b
        self.pbozs_nu[:, 1:] = boozmn.pmns_b
        self.gbozc_nu[:, 1:] = boozmn.gmnc_b

        if boozmn.lasym_b:
            self.bbozs_nu[:, 1:] = boozmn.bmns_b
            self.rbozs_nu[:, 1:] = boozmn.rmns_b
            self.zbozc_nu[:, 1:] = boozmn.zmnc_b
            self.pbozc_nu[:, 1:] = boozmn.pmnc_b
            self.gbozs_nu[:, 1:] = boozmn.gmns_b

        # extrapolation
        for imn in range(boozmn.mnboz_b):
            if boozmn.ixm_b[imn] == 0:
                self.bbozc_nu[imn, 0] = (3.0 * self.bbozc_nu[imn, 1] -
                                         3.0 * self.bbozc_nu[imn, 2] + self.bbozc_nu[imn, 3])
                self.rbozc_nu[imn, 0] = (3.0 * self.rbozc_nu[imn, 1] -
                                         3.0 * self.rbozc_nu[imn, 2] + self.rbozc_nu[imn, 3])
                self.zbozs_nu[imn, 0] = (3.0 * self.zbozs_nu[imn, 1] -
                                         3.0 * self.zbozs_nu[imn, 2] + self.zbozs_nu[imn, 3])
                self.pbozs_nu[imn, 0] = (3.0 * self.pbozs_nu[imn, 1] -
                                         3.0 * self.pbozs_nu[imn, 2] + self.pbozs_nu[imn, 3])
                self.gbozc_nu[imn, 0] = (3.0 * self.gbozc_nu[imn, 1] -
                                         3.0 * self.gbozc_nu[imn, 2] + self.gbozc_nu[imn, 3])
            else:
                self.bbozc_nu[imn, 0] = 0.0
                self.rbozc_nu[imn, 0] = 0.0
                self.zbozs_nu[imn, 0] = 0.0
                self.pbozs_nu[imn, 0] = 0.0
                self.gbozc_nu[imn, 0] = 0.0

            if boozmn.lasym_b:
                for imn in range(boozmn.mnboz_b):
                    if boozmn.ixm_b[imn] == 0:
                        self.bbozs_nu[imn, 0] = (3.0 * self.bbozs_nu[imn, 1] -
                                                 3.0 * self.bbozs_nu[imn, 2] + self.bbozs_nu[imn, 3])
                        self.rbozs_nu[imn, 0] = (3.0 * self.rbozs_nu[imn, 1] -
                                                 3.0 * self.rbozs_nu[imn, 2] + self.rbozs_nu[imn, 3])
                        self.zbozc_nu[imn, 0] = (3.0 * self.zbozc_nu[imn, 1] -
                                                 3.0 * self.zbozc_nu[imn, 2] + self.zbozc_nu[imn, 3])
                        self.pbozc_nu[imn, 0] = (3.0 * self.pbozc_nu[imn, 1] -
                                                 3.0 * self.pbozc_nu[imn, 2] + self.pbozc_nu[imn, 3])
                        self.gbozs_nu[imn, 0] = (3.0 * self.gbozs_nu[imn, 1] -
                                                 3.0 * self.gbozs_nu[imn, 2] + self.gbozs_nu[imn, 3])
                    else:
                        self.bbozs_nu[imn, 0] = 0.0
                        self.rbozs_nu[imn, 0] = 0.0
                        self.zbozc_nu[imn, 0] = 0.0
                        self.pbozc_nu[imn, 0] = 0.0
                        self.gbozs_nu[imn, 0] = 0.0

        boozmn.phi_b_nu[0] = 0.0
        boozmn.iota_b_nu[0] = (3.0 * boozmn.iota_b_nu[1] -
                               3.0 * boozmn.iota_b_nu[2] + boozmn.iota_b_nu[3])
        boozmn.bvco_b_nu[0] = 0.0
        boozmn.buco_b_nu[0] = (3.0 * boozmn.buco_b_nu[1] -
                               3.0 * boozmn.buco_b_nu[2] + boozmn.buco_b_nu[3])
        boozmn.pres_b_nu[0] = (3.0 * boozmn.pres_b_nu[1] -
                               3.0 * boozmn.pres_b_nu[2] + boozmn.pres_b_nu[3])
        boozmn.beta_b_nu[0] = (3.0 * boozmn.beta_b_nu[1] -
                               3.0 * boozmn.beta_b_nu[2] + boozmn.beta_b_nu[3])
        boozmn.phip_b_nu[0] = 0.0

        boozmn.phi_b_nu /= (2.0 * numpy.pi)
        boozmn.phip_b_nu /= (2.0 * numpy.pi)

    def normalization(self, boozmn: Boozmn, B0_p: float, Rmajor_p: float):
        # --- normalization for vmec calculation and (Bax, Rax, a) for GKV
        # rescale factor --> cnorm*bbozc (not used ordinary)
        cnorm = Rmajor_p * B0_p / numpy.abs(boozmn.bvco_b_nu[boozmn.ns_b - 1])
        print(f'cnorm: {cnorm}')

        self.Bax = self.bbozc_nu[0, 0]
        self.Rax = self.rbozc_nu[0, 0]
        self.aa = numpy.sqrt(
            2.0 * abs(boozmn.phi_b_nu[boozmn.ns_b - 1]) / self.Bax)
        print("Bax, Rax, a (for GKV), Phi_edge = ", self.Bax,
              self.Rax, self.aa, boozmn.phi_b_nu[boozmn.ns_b - 1])
        print("Bax, Rax, a (for GKV), Phi_edge = ", self.Bax,
              self.Rax, self.aa, boozmn.phi_b_nu[boozmn.ns_b - 1],
              file=self.olog)

    def q_profile(self, Ntheta_gkv: int, nrho: int, ntht: int, nzeta: int, boozmn: Boozmn):
        # --- rho & theta_b & zeta_b grids and q-profile
        self.rho = numpy.zeros(nrho)
        self.rho_nu = numpy.zeros(boozmn.ns_b)
        self.rho2 = numpy.zeros(boozmn.ns_b)
        self.qq_nu = numpy.zeros(boozmn.ns_b)

        self.theta = numpy.zeros(ntht + 1)
        self.zeta = numpy.zeros(nzeta + 1)

        for js in range(boozmn.ns_b):
            self.rho_nu[js] = numpy.sqrt(
                boozmn.phi_b_nu[js] / boozmn.phi_b_nu[boozmn.ns_b - 1])   # Non-Uniform rho grid
            self.rho2[js] = boozmn.phi_b_nu[js] / \
                boozmn.phi_b_nu[boozmn.ns_b - 1]

        print("rho_nu[0], rho_nu[1], iota_bar[0], iota_bar[1], = ")
        print(self.rho_nu[0], self.rho_nu[1],
              boozmn.iota_b_nu[0], boozmn.iota_b_nu[1])
        print("rho_nu[0], rho_nu[1], iota_bar[0], iota_bar[1], = ",
              file=self.olog)
        print(self.rho_nu[0], self.rho_nu[1],
              boozmn.iota_b_nu[0], boozmn.iota_b_nu[1],
              file=self.olog)

        for iz in range(nzeta + 1):
            for it in range(ntht + 1):
                self.theta[it] = float(Ntheta_gkv) * \
                    (2.0 * numpy.pi * it / ntht - numpy.pi)
                #%%% S.Mae 2025.3.8
                # self.zeta[iz] = 2.0 * numpy.pi * iz / nzeta
                self.zeta[iz] = numpy.divide(2.0 * numpy.pi * iz, nzeta, out=numpy.zeros_like(nzeta,dtype=float), where=(nzeta!=0))
                #%%%

        for js in range(nrho):
            self.rho[js] = js / numpy.abs(nrho - 1.0)       # Uniform rho grid

        for js in range(boozmn.ns_b):
            self.qq_nu[js] = 1.0 / numpy.abs(boozmn.iota_b_nu[js])

    def interpolation_to_uniform(self, nrho: int, boozmn: Boozmn):
        # --- interpolation to uniform rho-grids
        self.qq = numpy.zeros(nrho)
        self.dqdrho = numpy.zeros(nrho)
        self.shat = numpy.zeros(nrho)
        self.epst = numpy.zeros(nrho)

        self.cug = numpy.zeros(nrho)
        self.cui = numpy.zeros(nrho)
        self.dummy1 = numpy.zeros(nrho)
        self.phi_b = numpy.zeros(nrho)
        self.dphidrho = numpy.zeros(nrho)

        self.bbozc = numpy.zeros((boozmn.mnboz_b, nrho))
        self.rbozc = numpy.zeros((boozmn.mnboz_b, nrho))
        self.zbozs = numpy.zeros((boozmn.mnboz_b, nrho))
        self.pbozs = numpy.zeros((boozmn.mnboz_b, nrho))
        self.gbozc = numpy.zeros((boozmn.mnboz_b, nrho))

        self.dbbozc = numpy.zeros((boozmn.mnboz_b, nrho))
        self.drbozc = numpy.zeros((boozmn.mnboz_b, nrho))
        self.dzbozs = numpy.zeros((boozmn.mnboz_b, nrho))
        self.dpbozs = numpy.zeros((boozmn.mnboz_b, nrho))
        self.dummy2 = numpy.zeros((boozmn.mnboz_b, nrho))

        if boozmn.lasym_b:
            self.bbozs = numpy.zeros((boozmn.mnboz_b, nrho))
            self.rbozs = numpy.zeros((boozmn.mnboz_b, nrho))
            self.zbozc = numpy.zeros((boozmn.mnboz_b, nrho))
            self.pbozc = numpy.zeros((boozmn.mnboz_b, nrho))
            self.gbozs = numpy.zeros((boozmn.mnboz_b, nrho))

            self.dbbozs = numpy.zeros((boozmn.mnboz_b, nrho))
            self.drbozs = numpy.zeros((boozmn.mnboz_b, nrho))
            self.dzbozc = numpy.zeros((boozmn.mnboz_b, nrho))
            self.dpbozc = numpy.zeros((boozmn.mnboz_b, nrho))

        spline = Spline(boozmn.ns_b)

        spline.cubic_spline_pre(self.rho_nu, self.qq_nu, boozmn.ns_b)
        spline.cubic_spline_all(nrho, self.rho, self.qq, self.dqdrho)

        self.shat = self.dqdrho * self.rho / self.qq
        self.epst = self.rho * self.aa / self.Rax

        # --- cug: B_zeta  (covariant zeta comp. of B, or toroidal current func.)
        spline.cubic_spline_pre(self.rho_nu, boozmn.bvco_b_nu, boozmn.ns_b)
        spline.cubic_spline_all(nrho, self.rho, self.cug, self.dummy1)

        # --- cui: B_theta (covariant theta comp. of B, or poloidal current func.)
        spline.cubic_spline_pre(self.rho_nu, boozmn.buco_b_nu, boozmn.ns_b)
        spline.cubic_spline_all(nrho, self.rho, self.cui, self.dummy1)

        spline.cubic_spline_pre(self.rho_nu, boozmn.phi_b_nu, boozmn.ns_b)
        spline.cubic_spline_all(nrho, self.rho, self.phi_b, self.dphidrho)

        for imn in range(boozmn.mnboz_b):
            spline.cubic_spline_pre(
                self.rho_nu, self.bbozc_nu[imn, :], boozmn.ns_b)
            spline.cubic_spline_all(
                nrho, self.rho, self.bbozc[imn, :], self.dbbozc[imn, :])

        for imn in range(boozmn.mnboz_b):
            spline.cubic_spline_pre(
                self.rho_nu, self.rbozc_nu[imn, :], boozmn.ns_b)
            spline.cubic_spline_all(
                nrho, self.rho, self.rbozc[imn, :], self.drbozc[imn, :])

        for imn in range(boozmn.mnboz_b):
            spline.cubic_spline_pre(
                self.rho_nu, self.zbozs_nu[imn, :], boozmn.ns_b)
            spline.cubic_spline_all(
                nrho, self.rho, self.zbozs[imn, :], self.dzbozs[imn, :])

        for imn in range(boozmn.mnboz_b):
            spline.cubic_spline_pre(
                self.rho_nu, self.pbozs_nu[imn, :], boozmn.ns_b)
            spline.cubic_spline_all(
                nrho, self.rho, self.pbozs[imn, :], self.dpbozs[imn, :])

        # --- gbozc: rootg_boz/dphidrho
        for imn in range(boozmn.mnboz_b):
            spline.cubic_spline_pre(
                self.rho_nu, self.gbozc_nu[imn, :], boozmn.ns_b)
            spline.cubic_spline_all(
                nrho, self.rho, self.gbozc[imn, :], self.dummy2[imn, :])

        if boozmn.lasym_b:
            for imn in range(boozmn.mnboz_b):
                spline.cubic_spline_pre(
                    self.rho_nu, self.bbozs_nu[imn, :], boozmn.ns_b)
                spline.cubic_spline_all(
                    nrho, self.rho, self.bbozs[imn, :], self.dbbozs[imn, :])

            for imn in range(boozmn.mnboz_b):
                spline.cubic_spline_pre(
                    self.rho_nu, self.rbozs_nu[imn, :], boozmn.ns_b)
                spline.cubic_spline_all(
                    nrho, self.rho, self.rbozs[imn, :], self.drbozs[imn, :])

            for imn in range(boozmn.mnboz_b):
                spline.cubic_spline_pre(
                    self.rho_nu, self.zbozc_nu[imn, :], boozmn.ns_b)
                spline.cubic_spline_all(
                    nrho, self.rho, self.zbozc[imn, :], self.dzbozc[imn, :])

            for imn in range(boozmn.mnboz_b):
                spline.cubic_spline_pre(
                    self.rho_nu, self.pbozc_nu[imn, :], boozmn.ns_b)
                spline.cubic_spline_all(
                    nrho, self.rho, self.pbozc[imn, :], self.dpbozc[imn, :])

            # --- gbozc: rootg_boz/dphidrho
            for imn in range(boozmn.mnboz_b):
                spline.cubic_spline_pre(
                    self.rho_nu, self.gbozs_nu[imn, :], boozmn.ns_b)
                spline.cubic_spline_all(
                    nrho, self.rho, self.gbozs[imn, :], self.dummy2[imn, :])

    def B_R_Z_Phi(self, nrho: int, ntht: int, nzeta: int, alpha_fix: float, boozmn: Boozmn):
        self.bb = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.rr = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.zz = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.ph = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.ggb = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        for js in range(nrho):
            print(f'js-loop num. = {js}')
            print(f'js-loop num. = {js}', file=self.olog)

            for iz in range(nzeta + 1):
                for it in range(ntht + 1):
                    phases = numpy.zeros(boozmn.mnboz_b)

                    if nzeta == 0:
                        self.zeta[iz] = (self.qq[js] *
                                         self.theta[it] + alpha_fix)

                    phases = (-boozmn.ixn_b * self.zeta[iz] +
                              boozmn.ixm_b * self.theta[it])

                    cos_phases = numpy.cos(phases)
                    sin_phases = numpy.sin(phases)

                    self.bb[js, it, iz] += numpy.sum(
                        self.bbozc[:, js] * cos_phases)
                    self.rr[js, it, iz] += numpy.sum(
                        self.rbozc[:, js] * cos_phases)
                    self.zz[js, it, iz] += numpy.sum(
                        self.zbozs[:, js] * sin_phases)
                    self.ph[js, it, iz] += numpy.sum(
                        self.pbozs[:, js] * sin_phases)
                    self.ggb[js, it, iz] += numpy.sum(
                        self.gbozc[:, js] * cos_phases)
                    self.dbb_drho[js, it, iz] += numpy.sum(
                        self.dbbozc[:, js] * cos_phases)
                    self.drr_drho[js, it, iz] += numpy.sum(
                        self.drbozc[:, js] * cos_phases)
                    self.dzz_drho[js, it, iz] += numpy.sum(
                        self.dzbozs[:, js] * sin_phases)
                    self.dph_drho[js, it, iz] += numpy.sum(
                        self.dpbozs[:, js] * sin_phases)

                    self.dbb_dtht[js, it, iz] -= numpy.sum(
                        boozmn.ixm_b * self.bbozc[:, js] * sin_phases)
                    self.drr_dtht[js, it, iz] -= numpy.sum(
                        boozmn.ixm_b * self.rbozc[:, js] * sin_phases)
                    self.dzz_dtht[js, it, iz] += numpy.sum(
                        boozmn.ixm_b * self.zbozs[:, js] * cos_phases)
                    self.dph_dtht[js, it, iz] += numpy.sum(
                        boozmn.ixm_b * self.pbozs[:, js] * cos_phases)

                    self.dbb_dzeta[js, it, iz] += numpy.sum(
                        boozmn.ixn_b * self.bbozc[:, js] * sin_phases)
                    self.drr_dzeta[js, it, iz] += numpy.sum(
                        boozmn.ixn_b * self.rbozc[:, js] * sin_phases)
                    self.dzz_dzeta[js, it, iz] -= numpy.sum(
                        boozmn.ixn_b * self.zbozs[:, js] * cos_phases)
                    self.dph_dzeta[js, it, iz] -= numpy.sum(
                        boozmn.ixn_b * self.pbozs[:, js] * cos_phases)

                    self.ph[js, it, iz] = self.zeta[iz] + self.ph[js, it, iz]
                    self.dph_drho[js, it, iz] = self.dph_drho[js, it, iz]
                    self.dph_dtht[js, it, iz] = self.dph_dtht[js, it, iz]
                    self.dph_dzeta[js, it, iz] = 1.0 + \
                        self.dph_dzeta[js, it, iz]

    #%%% S.Mae 2025.3.8
    def B_R_Z_Phi_tune(self, nrho: int, ntht: int, nzeta: int, alpha_fix: float, boozmn: Boozmn):
        self.bb = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.rr = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.zz = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.ph = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.ggb = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_drho = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_dtht = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        self.dbb_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.drr_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dzz_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))
        self.dph_dzeta = numpy.zeros((nrho, ntht + 1, nzeta + 1))

        if nzeta == 0:
            self.zeta[0] = self.qq[-1]*self.theta[-1]+alpha_fix

            phases = numpy.zeros(boozmn.mnboz_b)
            wzeta = (self.qq[numpy.newaxis,:] * self.theta[:,numpy.newaxis] + alpha_fix) # itheta,js
            phases = (-boozmn.ixn_b[numpy.newaxis,numpy.newaxis,:] * wzeta[:,:,numpy.newaxis] 
                    + boozmn.ixm_b[numpy.newaxis,numpy.newaxis,:] * self.theta[:,numpy.newaxis,numpy.newaxis]) # itheta,js,imn

            cos_phases = numpy.cos(phases)
            sin_phases = numpy.sin(phases)

            self.bb[:,:,0]       += numpy.einsum("tsm,ms->ts", cos_phases, self.bbozc).T
            self.rr[:,:,0]       += numpy.einsum("tsm,ms->ts", cos_phases, self.rbozc).T
            self.zz[:,:,0]       += numpy.einsum("tsm,ms->ts", sin_phases, self.zbozs).T
            self.ph[:,:,0]       += numpy.einsum("tsm,ms->ts", sin_phases, self.pbozs).T
            self.ggb[:,:,0]      += numpy.einsum("tsm,ms->ts", cos_phases, self.gbozc).T
            self.dbb_drho[:,:,0] += numpy.einsum("tsm,ms->ts", cos_phases, self.dbbozc).T
            self.drr_drho[:,:,0] += numpy.einsum("tsm,ms->ts", cos_phases, self.drbozc).T
            self.dzz_drho[:,:,0] += numpy.einsum("tsm,ms->ts", sin_phases, self.dzbozs).T
            self.dph_drho[:,:,0] += numpy.einsum("tsm,ms->ts", sin_phases, self.dpbozs).T

            self.dbb_dtht[:,:,0] -= numpy.einsum("tsm,m,ms->ts", sin_phases, boozmn.ixm_b, self.bbozc).T
            self.drr_dtht[:,:,0] -= numpy.einsum("tsm,m,ms->ts", sin_phases, boozmn.ixm_b, self.rbozc).T
            self.dzz_dtht[:,:,0] += numpy.einsum("tsm,m,ms->ts", cos_phases, boozmn.ixm_b, self.zbozs).T
            self.dph_dtht[:,:,0] += numpy.einsum("tsm,m,ms->ts", cos_phases, boozmn.ixm_b, self.pbozs).T

            self.dbb_dzeta[:,:,0] += numpy.einsum("tsm,m,ms->ts", sin_phases, boozmn.ixn_b, self.bbozc).T
            self.drr_dzeta[:,:,0] += numpy.einsum("tsm,m,ms->ts", sin_phases, boozmn.ixn_b, self.rbozc).T
            self.dzz_dzeta[:,:,0] -= numpy.einsum("tsm,m,ms->ts", cos_phases, boozmn.ixn_b, self.zbozs).T
            self.dph_dzeta[:,:,0] -= numpy.einsum("tsm,m,ms->ts", cos_phases, boozmn.ixn_b, self.pbozs).T

            self.ph = wzeta.T[:,:,numpy.newaxis] + self.ph # js,itheta,izeta
            self.dph_drho = self.dph_drho
            self.dph_dtht = self.dph_dtht
            self.dph_dzeta = 1.0 + self.dph_dzeta
        else:
            phases = (-boozmn.ixn_b[numpy.newaxis,numpy.newaxis,:] * self.zeta[:,numpy.newaxis,numpy.newaxis] +
                    boozmn.ixm_b[numpy.newaxis,numpy.newaxis,:] * self.theta[numpy.newaxis,:,numpy.newaxis]) # izeta,itheta,imn

            cos_phases = numpy.cos(phases)
            sin_phases = numpy.sin(phases)

            self.bb       += numpy.einsum("ztm,ms->zts", cos_phases, self.bbozc).T
            self.rr       += numpy.einsum("ztm,ms->zts", cos_phases, self.rbozc).T
            self.zz       += numpy.einsum("ztm,ms->zts", sin_phases, self.zbozs).T
            self.ph       += numpy.einsum("ztm,ms->zts", sin_phases, self.pbozs).T
            self.ggb      += numpy.einsum("ztm,ms->zts", cos_phases, self.gbozc).T
            self.dbb_drho += numpy.einsum("ztm,ms->zts", cos_phases, self.dbbozc).T
            self.drr_drho += numpy.einsum("ztm,ms->zts", cos_phases, self.drbozc).T
            self.dzz_drho += numpy.einsum("ztm,ms->zts", sin_phases, self.dzbozs).T
            self.dph_drho += numpy.einsum("ztm,ms->zts", sin_phases, self.dpbozs).T

            self.dbb_dtht -= numpy.einsum("ztm,m,ms->zts", sin_phases, boozmn.ixm_b, self.bbozc).T
            self.drr_dtht -= numpy.einsum("ztm,m,ms->zts", sin_phases, boozmn.ixm_b, self.rbozc).T
            self.dzz_dtht += numpy.einsum("ztm,m,ms->zts", cos_phases, boozmn.ixm_b, self.zbozs).T
            self.dph_dtht += numpy.einsum("ztm,m,ms->zts", cos_phases, boozmn.ixm_b, self.pbozs).T

            self.dbb_dzeta += numpy.einsum("ztm,m,ms->zts", sin_phases, boozmn.ixn_b, self.bbozc).T
            self.drr_dzeta += numpy.einsum("ztm,m,ms->zts", sin_phases, boozmn.ixn_b, self.rbozc).T
            self.dzz_dzeta -= numpy.einsum("ztm,m,ms->zts", cos_phases, boozmn.ixn_b, self.zbozs).T
            self.dph_dzeta -= numpy.einsum("ztm,m,ms->zts", cos_phases, boozmn.ixn_b, self.pbozs).T

            self.ph = self.zeta[numpy.newaxis,numpy.newaxis,:] + self.ph # js,itheta,izeta
            self.dph_drho = self.dph_drho
            self.dph_dtht = self.dph_dtht
            self.dph_dzeta = 1.0 + self.dph_dzeta
    #%%%

    def metric_boozer(self, nss: int, ntht: int, nzeta: int):
        # --- calculation and output of metric tensor
        self.ggup_boz = numpy.zeros((nss, ntht + 1, nzeta + 1, 3, 3))
        self.ggdn_boz = numpy.zeros((nss, ntht + 1, nzeta + 1, 3, 3))
        self.ggsq_boz = numpy.zeros((nss, ntht + 1, nzeta + 1))
        self.rootg_boz = numpy.zeros((nss, ntht + 1, nzeta + 1))
        self.rootg_boz0 = numpy.zeros((nss, ntht + 1, nzeta + 1))
        self.rootg_boz1 = numpy.zeros((nss, ntht + 1, nzeta + 1))

        # Co-variant metric tensor
        self.ggdn_boz[:, :, :, 0, 0] = self.drr_drho ** 2 + \
            self.dzz_drho ** 2 + self.rr ** 2 * self.dph_drho ** 2
        self.ggdn_boz[:, :, :, 0, 1] = self.drr_drho * self.drr_dtht \
            + self.dzz_drho * self.dzz_dtht \
            + self.rr ** 2 * self.dph_drho * self.dph_dtht
        self.ggdn_boz[:, :, :, 0, 2] = self.drr_drho * self.drr_dzeta \
            + self.dzz_drho * self.dzz_dzeta \
            + self.rr ** 2 * self.dph_drho * self.dph_dzeta

        self.ggdn_boz[:, :, :, 1, 0] = self.ggdn_boz[:, :, :, 0, 1]
        self.ggdn_boz[:, :, :, 1, 1] = self.drr_dtht ** 2 + \
            self.dzz_dtht ** 2 + self.rr ** 2 * self.dph_dtht ** 2
        self.ggdn_boz[:, :, :, 1, 2] = self.drr_dtht * self.drr_dzeta \
            + self.dzz_dtht * self.dzz_dzeta \
            + self.rr ** 2 * self.dph_dtht * self.dph_dzeta

        self.ggdn_boz[:, :, :, 2, 0] = self.ggdn_boz[:, :, :, 0, 2]
        self.ggdn_boz[:, :, :, 2, 1] = self.ggdn_boz[:, :, :, 1, 2]
        self.ggdn_boz[:, :, :, 2, 2] = self.drr_dzeta ** 2 + \
            self.dzz_dzeta ** 2 + self.rr ** 2 * self.dph_dzeta ** 2

        for iz in range(nzeta + 1):
            for it in range(ntht + 1):
                for js in range(nss):
                    try:
                        # Co-variant metric tensor
                        self.ggdn_boz[js, it, iz, 0, 0] = (
                            self.drr_drho[js, it, iz] ** 2 +
                            self.dzz_drho[js, it, iz] ** 2 +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_drho[js, it, iz] ** 2)
                        self.ggdn_boz[js, it, iz, 0, 1] = (
                            self.drr_drho[js, it, iz] *
                            self.drr_dtht[js, it, iz] +
                            self.dzz_drho[js, it, iz] *
                            self.dzz_dtht[js, it, iz] +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_drho[js, it, iz] *
                            self.dph_dtht[js, it, iz])
                        self.ggdn_boz[js, it, iz, 0, 2] = (
                            self.drr_drho[js, it, iz] *
                            self.drr_dzeta[js, it, iz] +
                            self.dzz_drho[js, it, iz] *
                            self.dzz_dzeta[js, it, iz] +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_drho[js, it, iz] *
                            self.dph_dzeta[js, it, iz])

                        self.ggdn_boz[js, it, iz, 1, 0] = (
                            self.ggdn_boz[js, it, iz, 0, 1])
                        self.ggdn_boz[js, it, iz, 1, 1] = (
                            self.drr_dtht[js, it, iz] ** 2 +
                            self.dzz_dtht[js, it, iz] ** 2 +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_dtht[js, it, iz] ** 2)
                        self.ggdn_boz[js, it, iz, 1, 2] = (
                            self.drr_dtht[js, it, iz] *
                            self.drr_dzeta[js, it, iz] +
                            self.dzz_dtht[js, it, iz] *
                            self.dzz_dzeta[js, it, iz] +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_dtht[js, it, iz] *
                            self.dph_dzeta[js, it, iz])

                        self.ggdn_boz[js, it, iz, 2, 0] = (
                            self.ggdn_boz[js, it, iz, 0, 2])
                        self.ggdn_boz[js, it, iz, 2, 1] = (
                            self.ggdn_boz[js, it, iz, 1, 2])
                        self.ggdn_boz[js, it, iz, 2, 2] = (
                            self.drr_dzeta[js, it, iz] ** 2 +
                            self.dzz_dzeta[js, it, iz] ** 2 +
                            self.rr[js, it, iz] ** 2 *
                            self.dph_dzeta[js, it, iz] ** 2)

                        # Squared Jacobian
                        self.ggsq_boz[js, it, iz] = (self.ggdn_boz[js, it, iz, 0, 0] * self.ggdn_boz[js, it, iz, 1, 1] * self.ggdn_boz[js, it, iz, 2, 2] +
                                                     self.ggdn_boz[js, it, iz, 0, 1] * self.ggdn_boz[js, it, iz, 1, 2] * self.ggdn_boz[js, it, iz, 2, 0] +
                                                     self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 1, 0] * self.ggdn_boz[js, it, iz, 2, 1] -
                                                     self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 1, 1] * self.ggdn_boz[js, it, iz, 2, 0] -
                                                     self.ggdn_boz[js, it, iz, 0, 1] * self.ggdn_boz[js, it, iz, 1, 0] * self.ggdn_boz[js, it, iz, 2, 2] -
                                                     self.ggdn_boz[js, it, iz, 0, 0] * self.ggdn_boz[js, it, iz, 1, 2] * self.ggdn_boz[js, it, iz, 2, 1])

                        # Jacobian: rootg = sqrt(g)
                        self.rootg_boz[js, it, iz] = numpy.sqrt(
                            self.ggsq_boz[js, it, iz])
                        self.rootg_boz0[js, it, iz] = self.dphidrho[js] * (
                            self.cug[js] + self.cui[js] / self.qq[js]) / self.bb[js, it, iz] ** 2
                        self.rootg_boz1[js, it, iz] = (
                            self.dphidrho[js] * self.ggb[js, it, iz])

                        # Contra-variant metric tensor
                        self.ggup_boz[js, it, iz, 0, 0] = (self.ggdn_boz[js, it, iz, 1, 1] * self.ggdn_boz[js, it, iz, 2, 2] -
                                                           self.ggdn_boz[js, it, iz, 1, 2] * self.ggdn_boz[js, it, iz, 2, 1]) / self.ggsq_boz[js, it, iz]
                        self.ggup_boz[js, it, iz, 0, 1] = (self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 2, 1] -
                                                           self.ggdn_boz[js, it, iz, 0, 1] * self.ggdn_boz[js, it, iz, 2, 2]) / self.ggsq_boz[js, it, iz]
                        self.ggup_boz[js, it, iz, 0, 2] = (self.ggdn_boz[js, it, iz, 0, 1] * self.ggdn_boz[js, it, iz, 1, 2] -
                                                           self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 1, 1]) / self.ggsq_boz[js, it, iz]
                        self.ggup_boz[js, it, iz, 1, 0] = (
                            self.ggup_boz[js, it, iz, 0, 1])
                        self.ggup_boz[js, it, iz, 1, 1] = (self.ggdn_boz[js, it, iz, 0, 0] * self.ggdn_boz[js, it, iz, 2, 2] -
                                                           self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 2, 0]) / self.ggsq_boz[js, it, iz]
                        self.ggup_boz[js, it, iz, 1, 2] = (self.ggdn_boz[js, it, iz, 0, 2] * self.ggdn_boz[js, it, iz, 1, 0] -
                                                           self.ggdn_boz[js, it, iz, 0, 0] * self.ggdn_boz[js, it, iz, 1, 2]) / self.ggsq_boz[js, it, iz]
                        self.ggup_boz[js, it, iz, 2, 0] = (
                            self.ggup_boz[js, it, iz, 0, 2])
                        self.ggup_boz[js, it, iz, 2, 1] = (
                            self.ggup_boz[js, it, iz, 1, 2])
                        self.ggup_boz[js, it, iz, 2, 2] = (self.ggdn_boz[js, it, iz, 0, 0] * self.ggdn_boz[js, it, iz, 1, 1] -
                                                           self.ggdn_boz[js, it, iz, 0, 1] * self.ggdn_boz[js, it, iz, 1, 0]) / self.ggsq_boz[js, it, iz]
                    except ZeroDivisionError as e:
                        print(e)

    def output_metric(self, nss: int, ntht: int, nzeta: int, alpha_fix: float, volume_p,
                      file_name: str, boozmn: Boozmn, fortran_fomat=False):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #   All quantities have not been normalized for GKV here.
        #   Polidal angle range is [-pi:pi], but [0:2pi] for toroidal one.
        #
        #   List of output data:
        #             rho: radial coord.,                rho = sqrt(phi/phi_edge) [phi: toloidal flux / 2pi]
        #              qq: safety factor,                q(rho)
        #            shat: local magnetic shear,         s_hat(rho) = (rho/q)*dq/drho
        #            epst: local inverse aspect ratio,   eps(rho) = rho*a/Rax, a is defined as a = sqrt(2phi_edge/Bax)
        #             cug: covariant B-field comp.,      B_zeta(rho)  (or toroidal current function)
        #             cui: covariant B-field comp.,      B_theta(rho) (or poloidal current function)
        #        dphidrho: radial deriv of phi,          dphi/drho(rho)
        #              bb: B-field intensity,            B(rho,theta,zeta) [T]
        #              rr: B-field intensity,            R(rho,theta,zeta) [m]
        #              zz: B-field intensity,            Z(rho,theta,zeta) [m]
        #              ph: B-field intensity,            phi(rho,theta,zeta) in VMEC coord. [radian]
        #           dX_dY: derivatives in Booz. coord.,  dX/dY(rho,theta,zeta), X={bb,rr,zz,ph}, Y={rho,tht,zeta}
        #       rootg_boz: Jacobian,                     sqrt(g_boozer) [rho,theta,zeta]
        #            ggdn: covariant metric comp.,       g_ij [rho,theta,zeta] (i,j) = {rho,theta,zeta}
        #            ggup: contravariant metric comp.,   g^ij [rho,theta,zeta] (i,j) = {rho,theta,zeta}
        #           nfp_b: stellarator period            nfp_b = 10(LHD), = 5(W-7X), = 4(Heliotron-J) etc.
        #            mboz: the number of modes for m     poloidal mode number in Boozer coord. (specified by in_booz)
        #            nboz: the number of modes for n     toroidal mode number in Boozer coord. (specified by in_booz)
        #           mnboz: the total number of modes     mnboz = nboz+1 + (mboz-1)*(1+2*nboz)
        #           ixm_b: poloidal mode number          m(i), i=1, mnboz_b
        #           ixn_b: toroidal mode number          n(i), i=1, mnboz_b
        #           bbozc: Fourier spectrum of B         B_{n,m}(i), i=1, mnboz_b
        #             Bax: |B| at the axis               B_{m=0,n=0}(rho=0,theta=0,zeta=0) [T]
        #             Rax: |B| at the axis               R_{m=0,n=0}(rho=0,theta=0,zeta=0) [m]
        #              aa: plasma radius                 a = sqrt(2phi_edge/Bax)  [m]
        #        asym_flg: flag for up-down asym.        =1: up-down asymmetric fields, =0: up-down symmetric(default)
        #       alpha_fix: field line label              alpha_fix = zeta - q*theta
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # --- output of metric data in Boozer coord.
        # --- binary output
        asym_flg = 1 if boozmn.lasym_b else 0

        with scipy.io.FortranFile(file_name, 'w') as omtr:
            omtr.write_record(numpy.int32(boozmn.nfp_b), numpy.int32(nss), numpy.int32(ntht), numpy.int32(nzeta),
                              numpy.int32(boozmn.mnboz_b),
                              numpy.int32(boozmn.mboz_b),
                              numpy.int32(boozmn.nboz_b),
                              self.Rax, self.Bax, self.aa,
                              volume_p, numpy.int32(asym_flg), alpha_fix)
            omtr.write_record(self.rho, self.theta, self.zeta,
                              self.qq, self.shat, self.epst,
                              self.bb.T,
                              self.rootg_boz.T,
                              self.rootg_boz0.T,
                              self.ggup_boz.T,
                              self.dbb_drho.T,
                              self.dbb_dtht.T,
                              self.dbb_dzeta.T)
            omtr.write_record(self.rr.T,
                              self.zz.T,
                              self.ph.T,
                              self.bbozc.T,
                              boozmn.ixn_b.astype(numpy.int32),
                              boozmn.ixm_b.astype(numpy.int32))
            if boozmn.lasym_b:
                omtr.write_record(self.bbozs.T)

#%%% S.Mae 2025.3.9
    def output_mag_check(self, nss: int, ntht: int, nzeta: int, fortran_fomat=False):
        # --- for debug
        file_dbg = "./check/mag_check.dat"
        if fortran_fomat:
            fd = format_double  # alias for shorten name

            template = ("{:>24}" * 47) + '\n'

            with open(file_dbg, 'w') as odbg:
                for js in range(nss):
                    for iz in range(nzeta + 1):
                        for it in range(ntht + 1):
                            odbg.write(
                                template.format(
                                    fd(self.rho[js]),
                                    fd(self.theta[it]),
                                    fd(self.zeta[iz]),
                                    fd(self.qq[js]),
                                    fd(self.shat[js]),
                                    fd(self.epst[js]),
                                    fd(self.cui[js]),
                                    fd(self.cug[js]),
                                    fd(self.dphidrho[js]),
                                    fd(self.rootg_boz0[js, it, iz]),    # 10
                                    fd(self.rootg_boz1[js, it, iz]),    # 11
                                    fd(self.rootg_boz[js, it, iz]),  # 12
                                    fd(self.ggsq_boz[js, it, iz]),   # 13
                                    fd(self.ggdn_boz[js, it, iz, 0, 0]),  # 14
                                    fd(self.ggdn_boz[js, it, iz, 0, 1]),  # 15
                                    fd(self.ggdn_boz[js, it, iz, 0, 2]),  # 16
                                    fd(self.ggdn_boz[js, it, iz, 1, 0]),  # 17
                                    fd(self.ggdn_boz[js, it, iz, 1, 1]),  # 18
                                    fd(self.ggdn_boz[js, it, iz, 1, 2]),  # 19
                                    fd(self.ggdn_boz[js, it, iz, 2, 0]),  # 20
                                    fd(self.ggdn_boz[js, it, iz, 2, 1]),  # 21
                                    fd(self.ggdn_boz[js, it, iz, 2, 2]),  # 22
                                    fd(self.ggup_boz[js, it, iz, 0, 0]),  # 23
                                    fd(self.ggup_boz[js, it, iz, 0, 1]),  # 24
                                    fd(self.ggup_boz[js, it, iz, 0, 2]),  # 25
                                    fd(self.ggup_boz[js, it, iz, 1, 0]),  # 26
                                    fd(self.ggup_boz[js, it, iz, 1, 1]),  # 27
                                    fd(self.ggup_boz[js, it, iz, 1, 2]),  # 28
                                    fd(self.ggup_boz[js, it, iz, 2, 0]),  # 29
                                    fd(self.ggup_boz[js, it, iz, 2, 1]),  # 30
                                    fd(self.ggup_boz[js, it, iz, 2, 2]),  # 31
                                    fd(self.bb[js, it, iz]),  # 32
                                    fd(self.rr[js, it, iz]),  # 33
                                    fd(self.zz[js, it, iz]),  # 34
                                    fd(self.ph[js, it, iz]),  # 35
                                    fd(self.dbb_drho[js, it, iz]),  # 36
                                    fd(self.drr_drho[js, it, iz]),  # 37
                                    fd(self.dzz_drho[js, it, iz]),  # 38
                                    fd(self.dph_drho[js, it, iz]),  # 39
                                    fd(self.dbb_dtht[js, it, iz]),  # 40
                                    fd(self.drr_dtht[js, it, iz]),  # 41
                                    fd(self.dzz_dtht[js, it, iz]),  # 42
                                    fd(self.dph_dtht[js, it, iz]),  # 43
                                    fd(self.dbb_dzeta[js, it, iz]),  # 44
                                    fd(self.drr_dzeta[js, it, iz]),  # 45
                                    fd(self.dzz_dzeta[js, it, iz]),  # 46
                                    fd(self.dph_dzeta[js, it, iz])  # 47
                                ))

                    odbg.write('\n')
                odbg.write('\n')
        else:
            fmt = '%24.15E' * 47
            rho = numpy.zeros(ntht + 1)
            zeta = numpy.zeros(ntht + 1)
            qq = numpy.zeros(ntht + 1)
            shat = numpy.zeros(ntht + 1)
            epst = numpy.zeros(ntht + 1)
            cui = numpy.zeros(ntht + 1)
            cug = numpy.zeros(ntht + 1)
            dphidrho = numpy.zeros(ntht + 1)

            with open(file_dbg, 'w') as odbg:
                for js in range(nss):
                    for iz in range(nzeta + 1):
                        rho[:] = self.rho[js]
                        zeta[:] = self.zeta[iz]
                        qq[:] = self.qq[js]
                        shat[:] = self.shat[js]
                        epst[:] = self.epst[js]
                        cui[:] = self.cui[js]
                        cug[:] = self.cug[js]
                        dphidrho[:] = self.dphidrho[js]

                        numpy.savetxt(odbg, numpy.c_[
                            rho,
                            self.theta,
                            zeta,
                            qq,
                            shat,
                            epst,
                            cui,
                            cug,
                            dphidrho,
                            self.rootg_boz0[js, :, iz],    # 10
                            self.rootg_boz1[js, :, iz],    # 11
                            self.rootg_boz[js, :, iz],  # 12
                            self.ggsq_boz[js, :, iz],   # 13
                            self.ggdn_boz[js, :, iz, 0, 0],  # 14
                            self.ggdn_boz[js, :, iz, 0, 1],  # 15
                            self.ggdn_boz[js, :, iz, 0, 2],  # 16
                            self.ggdn_boz[js, :, iz, 1, 0],  # 17
                            self.ggdn_boz[js, :, iz, 1, 1],  # 18
                            self.ggdn_boz[js, :, iz, 1, 2],  # 19
                            self.ggdn_boz[js, :, iz, 2, 0],  # 20
                            self.ggdn_boz[js, :, iz, 2, 1],  # 21
                            self.ggdn_boz[js, :, iz, 2, 2],  # 22
                            self.ggup_boz[js, :, iz, 0, 0],  # 23
                            self.ggup_boz[js, :, iz, 0, 1],  # 24
                            self.ggup_boz[js, :, iz, 0, 2],  # 25
                            self.ggup_boz[js, :, iz, 1, 0],  # 26
                            self.ggup_boz[js, :, iz, 1, 1],  # 27
                            self.ggup_boz[js, :, iz, 1, 2],  # 28
                            self.ggup_boz[js, :, iz, 2, 0],  # 29
                            self.ggup_boz[js, :, iz, 2, 1],  # 30
                            self.ggup_boz[js, :, iz, 2, 2],  # 31
                            self.bb[js, :, iz],  # 32
                            self.rr[js, :, iz],  # 33
                            self.zz[js, :, iz],  # 34
                            self.ph[js, :, iz],  # 35
                            self.dbb_drho[js, :, iz],  # 36
                            self.drr_drho[js, :, iz],  # 37
                            self.dzz_drho[js, :, iz],  # 38
                            self.dph_drho[js, :, iz],  # 39
                            self.dbb_dtht[js, :, iz],  # 40
                            self.drr_dtht[js, :, iz],  # 41
                            self.dzz_dtht[js, :, iz],  # 42
                            self.dph_dtht[js, :, iz],  # 43
                            self.dbb_dzeta[js, :, iz],  # 44
                            self.drr_dzeta[js, :, iz],  # 45
                            self.dzz_dzeta[js, :, iz],  # 46
                            self.dph_dzeta[js, :, iz]  # 47
                        ], fmt=fmt)

                    odbg.write('\n')
                odbg.write('\n')
#%%%

    def output_geom(self, nrho: int,
                    ntht: int,
                    nzeta: int,
                    file_prf: str,
                    file_out1: str,
                    file_out2: str,
                    file_out3: str,
                    fortran_format=False):
        # --- output of geom data
        fd = format_double  # alias for shorten name

        with open(file_prf, 'w') as oprf:
            if fortran_format:
                template = "{:5}" + ("{:>24}" * 8) + '\n'

                for js in range(nrho):
                    oprf.write(
                        template.format(
                            js + 1,
                            fd(self.rho[js]),
                            fd(self.qq[js]),
                            fd(self.shat[js]),
                            fd(self.epst[js]),
                            fd(self.cug[js]),
                            fd(self.cui[js]),
                            fd(self.phi_b[js]),
                            fd(self.dphidrho[js])))
            else:
                fmt = '%5d' + ('%24.15E' * 8)
                jss = numpy.arange(1, nrho + 1)

                numpy.savetxt(oprf, numpy.c_[
                    jss,
                    self.rho,
                    self.qq,
                    self.shat,
                    self.epst,
                    self.cug,
                    self.cui,
                    self.phi_b,
                    self.dphidrho],
                    fmt=fmt)

        with open(file_out1, 'w') as oshp:
            if fortran_format:
                template = ("{:>24}" * 7) + '\n'

                for iz in range(nzeta + 1):
                    for it in range(ntht + 1):
                        for js in range(nrho):
                            oshp.write(
                                template.format(
                                    fd(self.rho[js]),
                                    fd(self.theta[it]),
                                    fd(self.zeta[iz]),
                                    fd(self.bb[js, it, iz]),
                                    fd(self.rr[js, it, iz]),
                                    fd(self.zz[js, it, iz]),
                                    fd(self.ph[js, it, iz])))
                        oshp.write("\n")
                    oshp.write("\n\n")
            else:
                fmt = '%24.15E' * 7

                theta = numpy.zeros(nrho)
                zeta = numpy.zeros(nrho)

                for iz in range(nzeta + 1):
                    for it in range(ntht + 1):
                        theta[:] = self.theta[it]
                        zeta[:] = self.zeta[iz]

                        numpy.savetxt(oshp, numpy.c_[
                            self.rho,
                            theta,
                            zeta,
                            self.bb[:, it, iz],
                            self.rr[:, it, iz],
                            self.zz[:, it, iz],
                            self.ph[:, it, iz]],
                            fmt=fmt)

                        oshp.write("\n")
                    oshp.write("\n\n")

        with open(file_out2, 'w') as oshp:
            if fortran_format:
                template = ("{:>24}" * 7) + '\n'

                for iz in range(nzeta + 1):
                    for js in range(nrho):
                        for it in range(ntht + 1):
                            oshp.write(
                                template.format(
                                    fd(self.rho[js]),
                                    fd(self.theta[it]),
                                    fd(self.zeta[iz]),
                                    fd(self.bb[js, it, iz]),
                                    fd(self.rr[js, it, iz]),
                                    fd(self.zz[js, it, iz]),
                                    fd(self.ph[js, it, iz])))
                        oshp.write("\n")
                    oshp.write("\n\n")
            else:
                fmt = '%24.15E' * 7
                rho = numpy.zeros(ntht + 1)
                zeta = numpy.zeros(ntht + 1)

                for iz in range(nzeta + 1):
                    for js in range(nrho):
                        rho[:] = self.rho[js]
                        zeta[:] = self.zeta[iz]

                        numpy.savetxt(oshp, numpy.c_[
                            rho,
                            self.theta,
                            zeta,
                            self.bb[js, :, iz],
                            self.rr[js, :, iz],
                            self.zz[js, :, iz],
                            self.ph[js, :, iz]],
                            fmt=fmt)

                        oshp.write("\n")
                    oshp.write("\n\n")

        with open(file_out3, 'w') as oshp:
            if fortran_format:
                template = ("{:>24}" * 7) + '\n'

                for js in range(nrho):
                    for iz in range(nzeta + 1):
                        for it in range(ntht + 1):
                            oshp.write(
                                template.format(
                                    fd(self.rho[js]),
                                    fd(self.theta[it]),
                                    fd(self.zeta[iz]),
                                    fd(self.bb[js, it, iz]),
                                    fd(self.rr[js, it, iz]),
                                    fd(self.zz[js, it, iz]),
                                    fd(self.ph[js, it, iz])))
                        oshp.write("\n")
                    oshp.write("\n\n")
            else:
                fmt = '%24.15E' * 7
                # reuse rho and zeta

                for js in range(nrho):
                    for iz in range(nzeta + 1):
                        rho[:] = self.rho[js]
                        zeta[:] = self.zeta[iz]

                        numpy.savetxt(oshp, numpy.c_[
                            rho,
                            self.theta,
                            zeta,
                            self.bb[js, :, iz],
                            self.rr[js, :, iz],
                            self.zz[js, :, iz],
                            self.ph[js, :, iz]],
                            fmt=fmt)

                        oshp.write("\n")
                    oshp.write("\n\n")

    def output_check(self,
                     file_check1: str,
                     file_check2: str,
                     file_check3: str,
                     boozmn: Boozmn,
                     fortran_format=False):
        # --- output for checking the read data
        fd = format_double  # alias for shorten name

        with open(file_check1, 'w') as odbg:
            if fortran_format:
                template = "{:5}" + ("{:>24}" * 10) + '\n'

                for js in range(boozmn.ns_b):
                    s = numpy.sqrt(
                        boozmn.phi_b_nu[js] / boozmn.phi_b_nu[boozmn.ns_b - 1])

                    odbg.write(
                        template.format(
                            js + 1,
                            fd(self.rho_nu[js]),
                            fd(boozmn.iota_b_nu[js]),
                            fd(self.qq_nu[js]),
                            fd(boozmn.pres_b_nu[js]),
                            fd(boozmn.beta_b_nu[js]),
                            fd(boozmn.phip_b_nu[js]),
                            fd(boozmn.phi_b_nu[js]),
                            fd(boozmn.bvco_b_nu[js]),
                            fd(boozmn.buco_b_nu[js]),
                            fd(s)))
            else:
                index = numpy.arange(1, boozmn.ns_b + 1)
                s = numpy.sqrt(
                    boozmn.phi_b_nu / boozmn.phi_b_nu[boozmn.ns_b - 1])
                fmt = '%5d' + ('%24.15E' * 10)

                numpy.savetxt(odbg, numpy.c_[
                              index,
                              self.rho_nu,
                              boozmn.iota_b_nu,
                              self.qq_nu,
                              boozmn.pres_b_nu,
                              boozmn.beta_b_nu,
                              boozmn.phip_b_nu,
                              boozmn.phi_b_nu,
                              boozmn.bvco_b_nu,
                              boozmn.buco_b_nu,
                              s],
                              fmt=fmt)

        with open(file_check2, 'w') as odbg:
            if fortran_format:
                if boozmn.lasym_b:
                    template = "{:5}{:5}" + ("{:>24}" * 10) + '\n'

                    for js in range(boozmn.ns_b - 1):
                        for imn in range(boozmn.mnboz_b):
                            odbg.write(
                                template.format(
                                    js + 1,
                                    imn + 1,
                                    fd(boozmn.bmnc_b[imn, js]),
                                    fd(boozmn.rmnc_b[imn, js]),
                                    fd(boozmn.zmns_b[imn, js]),
                                    fd(boozmn.pmns_b[imn, js]),
                                    fd(boozmn.gmnc_b[imn, js]),
                                    fd(boozmn.bmns_b[imn, js]),
                                    fd(boozmn.rmns_b[imn, js]),
                                    fd(boozmn.zmnc_b[imn, js]),
                                    fd(boozmn.pmnc_b[imn, js]),
                                    fd(boozmn.gmns_b[imn, js])))
                        odbg.write('\n')
                else:
                    template = "{:5}{:5}" + ("{:>24}" * 5) + '\n'

                    for js in range(boozmn.ns_b - 1):
                        for imn in range(boozmn.mnboz_b):
                            odbg.write(
                                template.format(
                                    js + 1,
                                    imn + 1,
                                    fd(boozmn.bmnc_b[imn, js]),
                                    fd(boozmn.rmnc_b[imn, js]),
                                    fd(boozmn.zmns_b[imn, js]),
                                    fd(boozmn.pmns_b[imn, js]),
                                    fd(boozmn.gmnc_b[imn, js])))
                        odbg.write('\n')
            else:
                imns = numpy.arange(1, boozmn.mnboz_b + 1)

                if boozmn.lasym_b:
                    fmt = '%5d%5d' + ('%24.15E' * 10)

                    for js in range(boozmn.ns_b - 1):
                        jss = numpy.ones(boozmn.mnboz_b) * (js + 1)

                        numpy.savetxt(odbg, numpy.c_[
                            jss,
                            imns,
                            boozmn.bmnc_b[:, js],
                            boozmn.rmnc_b[:, js],
                            boozmn.zmns_b[:, js],
                            boozmn.pmns_b[:, js],
                            boozmn.gmnc_b[:, js],
                            boozmn.bmns_b[:, js],
                            boozmn.rmns_b[:, js],
                            boozmn.zmnc_b[:, js],
                            boozmn.pmnc_b[:, js],
                            boozmn.gmns_b[:, js]],
                            fmt=fmt)

                        odbg.write('\n')
                else:
                    fmt = '%5d%5d' + ('%24.15E' * 5)

                    for js in range(boozmn.ns_b - 1):
                        jss = numpy.ones(boozmn.mnboz_b) * (js + 1)

                        numpy.savetxt(odbg, numpy.c_[
                            jss,
                            imns,
                            boozmn.bmnc_b[:, js],
                            boozmn.rmnc_b[:, js],
                            boozmn.zmns_b[:, js],
                            boozmn.pmns_b[:, js],
                            boozmn.gmnc_b[:, js]],
                            fmt=fmt)

                        odbg.write('\n')

        with open(file_check3, 'w') as odbg:
            if fortran_format:
                template = "{:5}{:5}{:5}{:>24}\n"

                for imn in range(boozmn.mnboz_b):
                    odbg.write(
                        template.format(
                            imn + 1,
                            boozmn.ixn_b[imn],
                            boozmn.ixm_b[imn],
                            fd(boozmn.bmnc_b[imn, (boozmn.ns_b - 1) // 2 - 1])))
            else:
                imns = numpy.arange(1, boozmn.mnboz_b + 1)

                numpy.savetxt(odbg, numpy.c_[
                    imns,
                    boozmn.ixn_b,
                    boozmn.ixm_b,
                    boozmn.bmnc_b[:, (boozmn.ns_b - 1) // 2 - 1]],
                    fmt='%5d%5d%5d%24.15E')


def input_from_boozmn(fname_boozmn: str) -> 'Boozmn':
    boozmn = Boozmn()

    #%%% S.Mae 2025.3.8
    if fname_boozmn[-3:] == ".nc": # NetCDF file-type "fname_wout"
        with (xr.load_dataset(fname_boozmn)) as ds:
    
            boozmn.nfp_b = int(ds["nfp_b"])
            boozmn.ns_b = int(ds["ns_b"])
            boozmn.aspect_b = float(ds["aspect_b"])
            boozmn.rmax_b = float(ds["rmax_b"])
            boozmn.rmin_b = float(ds["rmin_b"])
            boozmn.betaxis_b = float(ds["betaxis_b"])
        
            boozmn.iota_b_nu = ds["iota_b"].to_numpy()
            boozmn.pres_b_nu = ds["pres_b"].to_numpy()
            boozmn.beta_b_nu = ds["beta_b"].to_numpy()
            boozmn.phip_b_nu = ds["phip_b"].to_numpy()
            boozmn.phi_b_nu = ds["phi_b"].to_numpy()
            boozmn.bvco_b_nu = ds["bvco_b"].to_numpy()
            boozmn.buco_b_nu = ds["buco_b"].to_numpy()

            boozmn.mboz_b = int(ds["mboz_b"])
            boozmn.nboz_b = int(ds["nboz_b"])
            boozmn.mnboz_b = int(ds["mnboz_b"])
            boozmn.jsize = int(ds["jlist"].size)
        
            boozmn.version = ds["version"].item().decode()
        
            boozmn.lasym_b = bool(ds["lasym__logical__"])
        
            boozmn.ixn_b = ds["ixn_b"].to_numpy()
            boozmn.ixm_b = ds["ixm_b"].to_numpy()
        
            boozmn.jlist = ds["jlist"].to_numpy()
        
            boozmn.bmnc_b = ds["bmnc_b"].to_numpy().T
            boozmn.rmnc_b = ds["rmnc_b"].to_numpy().T
            boozmn.zmns_b = ds["zmns_b"].to_numpy().T
            boozmn.pmns_b = ds["pmns_b"].to_numpy().T
            boozmn.gmnc_b = ds["gmn_b"].to_numpy().T
        
            if boozmn.lasym_b:
                boozmn.bmns_b = ds["bmns_b"].to_numpy().T
                boozmn.rmns_b = ds["rmns_b"].to_numpy().T
                boozmn.zmnc_b = ds["zmnc_b"].to_numpy().T
                boozmn.pmnc_b = ds["pmnc_b"].to_numpy().T
                boozmn.gmns_b = ds["gmns_b"].to_numpy().T
        
    else: # BINARY file-type "fname_boozmn"

        with scipy.io.FortranFile(fname_boozmn, 'r') as f:
            try:
                boozmn.nfp_b, boozmn.ns_b, boozmn.aspect_b, boozmn.rmax_b, boozmn.rmin_b, boozmn.betaxis_b = f.read_record(
                    '<i4,<i4,<f8,<f8,<f8,<f8')[0]
            except:
                raise RuntimeError('Read error1 !!')
    
            boozmn.iota_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.pres_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.beta_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.phip_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.phi_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.bvco_b_nu = numpy.zeros(boozmn.ns_b)
            boozmn.buco_b_nu = numpy.zeros(boozmn.ns_b)
    
            try:
                for js in range(1, boozmn.ns_b):
                    boozmn.iota_b_nu[js], boozmn.pres_b_nu[js], boozmn.beta_b_nu[js], boozmn.phip_b_nu[js], boozmn.phi_b_nu[js], boozmn.bvco_b_nu[js], boozmn.buco_b_nu[js] = f.read_record(
                        '<f8')
            except:
                raise RuntimeError('Read error2 !!')
    
            try:
                boozmn.mboz_b, boozmn.nboz_b, boozmn.mnboz_b, boozmn.jsize = f.read_record(
                    'i4')
            except:
                raise RuntimeError('Read error3 !!')
    
            try:
                record = f.read_record('<u1')
                boozmn.version = ''.join(
                    [chr(c) for c in filter(lambda c: c != 0, record)]).strip()
                boozmn.lasym_b = record[-1] > 0
            except:
                raise RuntimeError('Read error4 !!')
    
            boozmn.bmnc_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
            boozmn.rmnc_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
            boozmn.zmns_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
            boozmn.pmns_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
            boozmn.gmnc_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
            boozmn.jlist = numpy.zeros(boozmn.ns_b - 1, dtype=numpy.int32)
    
            if boozmn.lasym_b:
                boozmn.bmns_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
                boozmn.rmns_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
                boozmn.zmnc_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
                boozmn.pmnc_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
                boozmn.gmns_b = numpy.zeros((boozmn.mnboz_b, boozmn.ns_b - 1))
    
            try:
                record = f.read_record(f'<i4')
                boozmn.ixn_b = record[: boozmn.mnboz_b]
                boozmn.ixm_b = record[boozmn.mnboz_b:]
            except:
                raise RuntimeError('Read error5 !!')
    
            for js in range(boozmn.jsize):  # jsize = ns_b - 1
                try:
                    record = f.read_record(f'<i4')
                    boozmn.jlist[js] = record[0]
                except:
                    raise RuntimeError('Read error6 !!')
    
                if (js > boozmn.ns_b or js <= -1):  # sic
                    continue
    
                try:
                    record = f.read_record(f'<f8')
                    boozmn.bmnc_b[:, js] = record[: boozmn.mnboz_b]
                    boozmn.rmnc_b[:, js] = record[
                        boozmn.mnboz_b: boozmn.mnboz_b * 2]
                    boozmn.zmns_b[:, js] = record[
                        boozmn.mnboz_b * 2: boozmn.mnboz_b * 3]
                    boozmn.pmns_b[:, js] = record[
                        boozmn.mnboz_b * 3: boozmn.mnboz_b * 4]
                    boozmn.gmnc_b[:, js] = record[boozmn.mnboz_b * 4:]
                except:
                    raise RuntimeError('Read error7 !!')
    
                try:
                    if boozmn.lasym_b:
                        record = f.read_record(f'<f8')
                        boozmn.bmns_b[:, js] = record[: boozmn.mnboz_b]
                        boozmn.rmns_b[:, js] = record[
                            boozmn.mnboz_b: boozmn.mnboz_b * 2]
                        boozmn.zmnc_b[:, js] = record[
                            boozmn.mnboz_b * 2: boozmn.mnboz_b * 3]
                        boozmn.pmnc_b[:, js] = record[
                            boozmn.mnboz_b * 3: boozmn.mnboz_b * 4]
                        boozmn.gmns_b[:, js] = record[
                            boozmn.mnboz_b * 4:]
                except:
                    raise RuntimeError('Read error8 !!')
    #%%%

    return boozmn


def print_boozmn(
        nrho: int, ntht: int, nzeta: int, alpha_fix: float,
        boozmn: 'Boozmn', fname_tag: str, file=sys.stdout):
    print(boozmn.version, file=file)
    print('# nfp_b, ns_b, aspect_b, rmax_b, rmin_b, betaxis_b = ', file=file)
    print(f'{boozmn.nfp_b} {boozmn.ns_b} {boozmn.aspect_b} {boozmn.rmax_b} '
          f'{boozmn.rmin_b} {boozmn.betaxis_b}', file=file)
    print('# mboz_b, nboz_b, mnboz_b, jsize, lasym_b = ', file=file)
    print(f'{boozmn.mboz_b} {boozmn.nboz_b} {boozmn.mnboz_b} {boozmn.jsize} {boozmn.lasym_b}',
          file=file)
    print(f'# fname_tag = {fname_tag}', file=file)

    if nzeta == 0:
        print('# nrho, ntheta, nzeta, alpha_fix = ', file=file)
        print(f'{nrho} {ntht} {nzeta} {alpha_fix}', file=file)
    else:
        print('# nrho, ntheta, nzeta = ', file=file)
        print(f'{nrho} {ntht} {nzeta}', file=file)

    if boozmn.lasym_b:
        print(' *** up-down asymmetric configuration *** ', file=file)
    else:
        print(' *** up-down symmetric configuration *** ', file=file)


def read_text(fname_wout: str, wout_txt: str) -> tuple[numpy.float64]:
    #%%% S.Mae 2025.3.8
    if fname_wout[-3:] == ".nc": # NetCDF file-type "fname_wout"
        with (xr.open_dataset(fname_wout)) as ds:
            B0_p = ds["b0"].to_numpy()
            Aminor_p = ds["Aminor_p"].to_numpy()
            Rmajor_p = ds["Rmajor_p"].to_numpy()
            volume_p = ds["volume_p"].to_numpy()
            return (B0_p, Aminor_p, Rmajor_p, volume_p)
    
    else: # ASCII file-type "fname_wout" 
        if not wout_txt:
            raise ValueError(f"wout_txt must be specified for ASCII files: {fname_wout}")

        with (open(fname_wout, 'r')) as f:
            while f:
                line = f.readline()
                if not line:
                    break
                if wout_txt in line:
                    break

            if not line:
                raise ValueError(f"Specified wout_txt '{wout_txt}' not found in {fname_wout}")
    
            if f:
                parameters = []
                while f:
                    line = f.readline()
                    parameters.extend(numpy.fromstring(
                        line, dtype=numpy.float64, sep=' '))
    
                    if len(parameters) >= 8:
                        break
    
                if len(parameters) >= 8:
                    B0_p = parameters[1]
                    Aminor_p = parameters[5]
                    Rmajor_p = parameters[6]
                    volume_p = parameters[7]
    
                    return (B0_p, Aminor_p, Rmajor_p, volume_p)
                else:
                    return None
    #%%%
    
    return None


def print_text(B0_p: float, Aminor_p: float, Rmajor_p: float, volume_p: float,
               file=sys.stdout):
    print('data point in wout found.', file=file)
    print('B0_p, Aminor_p, Rmajor_p, volume_p = ', file=file)
    print(f'{B0_p} {Aminor_p} {Rmajor_p} {volume_p}', file=file)


#%%% S. Mae 2025.3.10
def read_GKV_metric_file(fname_metric="./metric_boozer.bin.dat"):

    def detect_endianness(file_path):
        with open(file_path, "rb") as f:
            offset_header = 0
            record_length = (4*7)+(8*4)+4+8
            offset_footer = 4+record_length

            # Test big endian
            f.seek(offset_header)
            header = numpy.fromfile(f,dtype='>i4',count=1)
            f.seek(offset_footer)
            footer = numpy.fromfile(f,dtype='>i4',count=1)
            if not header.size:
                raise ValueError(f"Empty file: {file_path}")
            if header[0] == record_length and footer[0] == record_length:
                return ">" # Big endian

            # Test little endian
            f.seek(offset_header)
            header = numpy.fromfile(f,dtype='<i4',count=1)
            f.seek(offset_footer)
            footer = numpy.fromfile(f,dtype='<i4',count=1)
            if not header.size:
                raise ValueError("Empty file: {}".format(file_path))
            if header[0] == record_length and footer[0] == record_length:
                return "<" # Little endian
        raise ValueError(f"Failed to determine endianness: {file_path}")

    endianness = detect_endianness(fname_metric)
    print(f"Endianness check of {fname_metric}: {endianness}")
    dtype_i4 = numpy.dtype(f"{endianness}i4")  # 32bit int
    dtype_f8 = numpy.dtype(f"{endianness}f8")  # 64bit float

    with open(fname_metric, 'rb') as f:
        header = numpy.fromfile(f,dtype=dtype_i4,count=1)
        nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b = numpy.fromfile(f,dtype=dtype_i4,count=7)
        Rax, Bax, aa, volume_p = numpy.fromfile(f,dtype=dtype_f8,count=4)
        asym_flg = numpy.fromfile(f,dtype=dtype_i4,count=1)
        alpha_fix = numpy.fromfile(f,dtype=dtype_f8,count=1)
        footer = numpy.fromfile(f,dtype=dtype_i4,count=1)
        if header != footer:
            print("Error 1 for reading file_name:", fname_metric)

        header = numpy.fromfile(f,dtype=dtype_i4,count=1)
        rho = numpy.fromfile(f,dtype=dtype_f8,count=nss)
        theta = numpy.fromfile(f,dtype=dtype_f8,count=ntht+1)
        zeta = numpy.fromfile(f,dtype=dtype_f8,count=nzeta+1)
        qq = numpy.fromfile(f,dtype=dtype_f8,count=nss)
        shat = numpy.fromfile(f,dtype=dtype_f8,count=nss)
        epst = numpy.fromfile(f,dtype=dtype_f8,count=nss)
        bb = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        rootg_boz = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        rootg_boz0 = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        ggup_boz = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)*3*3).reshape((3,3,nzeta+1,ntht+1,nss)).T
        dbb_drho = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        dbb_dtht = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        dbb_dzeta = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        footer = numpy.fromfile(f,dtype=dtype_i4,count=1)
        if header != footer:
            print("Error 2 for reading file_name:", fname_metric)

        header = numpy.fromfile(f,dtype=dtype_i4,count=1)
        rr = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        zz = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        ph = numpy.fromfile(f,dtype=dtype_f8,count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        bbozc = numpy.fromfile(f,dtype=dtype_f8,count=mnboz_b*nss).reshape((nss,mnboz_b)).T
        ixn_b = numpy.fromfile(f,dtype=dtype_i4,count=mnboz_b)
        ixm_b = numpy.fromfile(f,dtype=dtype_i4,count=mnboz_b)
        footer = numpy.fromfile(f,dtype=dtype_i4,count=1)
        if header != footer:
            print("Error 3 for reading file_name:", fname_metric)

        if asym_flg == 1:
            header = numpy.fromfile(f,dtype=dtype_i4,count=1)
            bbozs = numpy.fromfile(f,dtype=dtype_f8,count=mnboz_b*nss).reshape((nss,mnboz_b)).T
            footer = numpy.fromfile(f,dtype=dtype_i4,count=1)
            if header != footer:
                print("Error 4 for reading file_name:", fname_metric)
        else:
            bbozs = None

    return (nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b, Rax, Bax, aa, volume_p, asym_flg, alpha_fix,
            rho, theta, zeta, qq, shat, epst, bb, rootg_boz, rootg_boz0, ggup_boz,
            dbb_drho, dbb_dtht, dbb_dzeta,
            rr, zz, ph, bbozc, ixn_b, ixm_b,
            bbozs)
#%%%

#def main():
#    file_log = './log_BZX.dat'
#    file_prf = './geom/prof.dat'
#    file_out1 = './geom/shape_rtz.dat'
#    file_out2 = './geom/shape_trz.dat'
#    file_out3 = './geom/shape_tzr.dat'
#    file_out4 = './geom/metric_boozer.bin.dat'
#
#    Ntheta_gkv: int = 1  # N_tht value in GKV
#    nrho: int = 11  # radial grid number in [0 <= rho <= 1]
#    # poloidal grid number in [-N_theta*pi < theta < N_theta*pi]
#    ntht: int = 64
#    nzeta: int = 64  # toroidal grid number in [0 <= zeta <= 2*pi]
#    alpha_fix: float = 0.0  # field-line label: alpha = zeta - q*theta NOT USED in 3d case
#
#    fname_tag: str = 'vmec_inward'
#    fname_boozmn: str = f'boozmn.{fname_tag}'
#    fname_wout: str = f'wout_{fname_tag}.txt'
#    wout_txt: str = fname_tag
#
#    with open(file_log, 'w') as f:
#        try:
#            boozmn = input_from_boozmn(fname_boozmn)
#            print_boozmn(nrho, ntht, nzeta, alpha_fix, boozmn, fname_tag)
#            print_boozmn(nrho, ntht, nzeta, alpha_fix, boozmn, fname_tag, f)
#
#            wout_parameters = read_text(fname_wout, wout_txt)
#            if wout_parameters is None:
#                print('file end')
#                sys.exit(1)
#
#            print_text(*wout_parameters, file=sys.stdout)
#            print_text(*wout_parameters, file=f)
#
#            B0_p: numpy.float64
#            Aminor_p: numpy.float64
#            Rmajor_p: numpy.float64
#            volume_p: numpy.float64
#
#            B0_p, Aminor_p, Rmajor_p, volume_p = wout_parameters
#
#            ##
#            metric = Metric(f)
#            metric.extrapolation_to_magnetic(boozmn)
#            metric.normalization(boozmn, B0_p, Rmajor_p)
#            metric.q_profile(Ntheta_gkv, nrho, ntht, nzeta, boozmn)
#            metric.interpolation_to_uniform(nrho, boozmn)
#            #%%% S.Mae 2025.3.8
#            # metric.B_R_Z_Phi(nrho, ntht, nzeta, alpha_fix, boozmn)
#            metric.B_R_Z_Phi_tune(nrho, ntht, nzeta, alpha_fix, boozmn)
#            #%%%
#            metric.metric_boozer(nrho, ntht, nzeta)
#
#            os.makedirs('check', exist_ok=True)
#            os.makedirs('geom', exist_ok=True)
#
#            metric.output_metric(nrho, ntht, nzeta, alpha_fix,
#                                 volume_p, file_out4, boozmn)
#
#            metric.output_geom(nrho, ntht, nzeta, file_prf,
#                               file_out1=file_out1,
#                               file_out2=file_out2,
#                               file_out3=file_out3)
#
#            file_check1 = './check/check1.dat'
#            file_check2 = './check/check2.dat'
#            file_check3 = './check/check3.dat'
#            metric.output_check(file_check1=file_check1,
#                                file_check2=file_check2,
#                                file_check3=file_check3,
#                                boozmn=boozmn)
#
#            print("DONE!!")
#            f.write("DONE!!")
#        except RuntimeError as e:
#            print(e)
#            print(e, file=f)
#
#
#if __name__ == '__main__':
#    main()


#%%% S. Mae 2025.3.8
def bzx(Ntheta_gkv: int, nrho: int, ntht: int, nzeta: int, alpha_fix: float = 0.0, fname_boozmn="boozmn.nc", fname_wout="wout.nc", output_file="./geom/metric_boozer.bin.dat", wout_txt: str = "", check_output: bool = False):
    file_log = './log_BZX.dat'
    file_prf = './geom/prof.dat'
    file_out1 = './geom/shape_rtz.dat'
    file_out2 = './geom/shape_trz.dat'
    file_out3 = './geom/shape_tzr.dat'
    file_out4 = output_file

    fname_tag = wout_txt
    
    with open(file_log, 'w') as f:
        try:
            boozmn = input_from_boozmn(fname_boozmn)
            print_boozmn(nrho, ntht, nzeta, alpha_fix, boozmn, fname_tag)
            print_boozmn(nrho, ntht, nzeta, alpha_fix, boozmn, fname_tag, f)

            wout_parameters = read_text(fname_wout, wout_txt)
            if wout_parameters is None:
                print('file end')
                sys.exit(1)

            print_text(*wout_parameters, file=sys.stdout)
            print_text(*wout_parameters, file=f)

            B0_p: numpy.float64
            Aminor_p: numpy.float64
            Rmajor_p: numpy.float64
            volume_p: numpy.float64

            B0_p, Aminor_p, Rmajor_p, volume_p = wout_parameters


            
            
            ##
            metric = Metric(f)
            metric.extrapolation_to_magnetic(boozmn)
            metric.normalization(boozmn, B0_p, Rmajor_p)
            metric.q_profile(Ntheta_gkv, nrho, ntht, nzeta, boozmn)
            metric.interpolation_to_uniform(nrho, boozmn)
            #%%% S.Mae 2025.3.8
            # metric.B_R_Z_Phi(nrho, ntht, nzeta, alpha_fix, boozmn)
            metric.B_R_Z_Phi_tune(nrho, ntht, nzeta, alpha_fix, boozmn)
            #%%%
            metric.metric_boozer(nrho, ntht, nzeta)

            metric.output_metric(nrho, ntht, nzeta, alpha_fix,
                                 volume_p, file_out4, boozmn)

            if check_output:
                os.makedirs('check', exist_ok=True)
                os.makedirs('geom', exist_ok=True)
                metric.output_mag_check(nrho, ntht, nzeta)
                metric.output_geom(nrho, ntht, nzeta, file_prf,
                                   file_out1=file_out1,
                                   file_out2=file_out2,
                                   file_out3=file_out3)

                file_check1 = './check/check1.dat'
                file_check2 = './check/check2.dat'
                file_check3 = './check/check3.dat'
                metric.output_check(file_check1=file_check1,
                                    file_check2=file_check2,
                                    file_check3=file_check3,
                                    boozmn=boozmn)

            print("DONE!!")
            f.write("DONE!!")
            
            #! --- extrapolation to the magnetic axis
        except RuntimeError as e:
            print(e)
            print(e, file=f)

# def main():
#     Ntheta_gkv: int = 1  
#     nrho: int = 11  # radial grid number in [0 <= rho <= 1]
#     ## poloidal grid number in [-N_theta*pi < theta < N_theta*pi]
#     ntht: int = 64
#     nzeta: int = 64  # toroidal grid number in [0 <= zeta <= 2*pi]
#     alpha_fix: float = 0.0  # field-line label: alpha = zeta - q*theta NOT USED in 3d case

#     fname_tag: str = 'vmec_inward'
#     fname_boozmn: str = f'boozmn.{fname_tag}'
#     fname_wout: str = f'wout_{fname_tag}.txt'
#     wout_txt: str = fname_tag

#     bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn, fname_wout, wout_txt, output_file="./geom/metric_boozer.bin.dat")

def main():
    import argparse
    # Default values for bzx function
    default_args = {
        "Ntheta_gkv": 1,
        "nrho": 11,
        "ntht": 64,
        "nzeta": 64,
        "alpha_fix": 0.0,
        "fname_boozmn": "boozmn.nc",
        "fname_wout": "wout.nc",
        "output_file": "./geom/metric_boozer.bin.dat",
        "wout_txt": "",
        "check_output": False
    }

    # If no command-line arguments are provided, run with default values
    if len(sys.argv) == 1:
        print("No command-line arguments provided. Running with default values...")
        bzx(**default_args)
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            description="Run BZX with command-line arguments",
            prog="python -m bzx"
            )

    parser.add_argument("--Ntheta_gkv", type=int, required=True, help="N_theta value in GKV")
    parser.add_argument("--nrho", type=int, required=True, help="Radial grid number in [0 <= rho <= 1]")
    parser.add_argument("--ntht", type=int, required=True, help="Poloidal grid number in [-N_theta*pi < theta < N_theta*pi]")
    parser.add_argument("--nzeta", type=int, required=True, help="Toroidal grid number in [0 <= zeta <= 2*pi]")
    parser.add_argument("--alpha_fix", type=float, required=True, help="Field-line label: alpha = zeta - q*theta")
    parser.add_argument("--fname_boozmn", type=str, required=True, help="Input Boozmn filename")
    parser.add_argument("--fname_wout", type=str, required=True, help="Input wout filename")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--wout_txt", type=str, help="Tag for wout file")
    parser.add_argument("--check_output", action="store_true", help="Enable additional output files for checking in check/ and geom/")

    # If arguments are missing, argparse will automatically print an error and exit
    args = parser.parse_args()

    # Execute bzx function with parsed arguments
    bzx(
        Ntheta_gkv=args.Ntheta_gkv,
        nrho=args.nrho,
        ntht=args.ntht,
        nzeta=args.nzeta,
        alpha_fix=args.alpha_fix,
        fname_boozmn=args.fname_boozmn,
        fname_wout=args.fname_wout,
        output_file=args.output_file,
        wout_txt=args.wout_txt,
        check_output=args.check_output
    )

if __name__ == '__main__':
    flag_profiling = False

    if flag_profiling:
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable() # Start of profiling region
        main()
        profiler.disable() # End of profiling region
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)

    else: 
        main()
# #%%%

