#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bzx import read_GKV_metric_file

fname_metric = "./metric_boozer.bin.dat"
nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b, Rax, Bax, aa, volume_p, asym_flg, alpha_fix, \
    rho, theta, zeta, qq, shat, epst, bb, rootg_boz, rootg_boz0, ggup_boz, \
    dbb_drho, dbb_dtht, dbb_dzeta, \
    rr, zz, ph, bbozc, ixn_b, ixm_b, \
    bbozs = read_GKV_metric_file(fname_metric)

print(nfp_b,nss,ntht,nzeta,bb.shape,theta.max(),zeta.max())
print("磁場強度 (B0):", Bax)
print("小半径 (Aminor_p):", aa)
print("大半径 (Rmajor_p):", Rax)
print("プラズマ体積 (volume_p):", volume_p)


# In[ ]:


assert nzeta == 0, "ERROR: plot_metric_nzeta_0.py assumes nzeta==0."


# ### (i) Radial profile

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(rho,qq,".-",label=r"Safety factor $q$")
ax.plot(rho,shat,"x-",label=r"Magnetic shear $\hat{s}$")
ax.plot(rho,epst,"*-",label=r"Inverse aspect ratio $\epsilon_r$")
ax.set_xlabel(r"Normalized minor radius $\rho$")
ax.set_ylabel(r"$\rho$, $\hat{s}$, $\epsilon_r$")
ax.legend()
plt.show()


# ### (ii) Metrics along a field line

# In[ ]:


js = nss // 2

data_dicts = {r"Geometrical toroidal angle $\varphi$" : ph[js,:,0],
              r"Magnetic field strength $B$"    : bb[js,:,0],
              r"$\partial B/\partial \rho$"     : dbb_drho[js,:,0],
              r"$\partial B/\partial \theta_B$" : dbb_dtht[js,:,0],
              r"$\partial B/\partial \zeta_B$"  : dbb_dzeta[js,:,0],
              r"Boozer coordinate Jacobian $\sqrt{g}_B$" : rootg_boz[js,:,0],
              r"Metric $g^{\rho,\rho}$"         : ggup_boz[js,:,0,0,0],
              r"Metric $g^{\rho,\theta}$"       : ggup_boz[js,:,0,0,1],
              # r"Metric $g^{\theta,\rho}$"       : ggup_boz[js,:,0,1,0],
              r"Metric $g^{\rho,\zeta}$"        : ggup_boz[js,:,0,0,2],
              # r"Metric $g^{\zeta,\rho}$"        : ggup_boz[js,:,0,2,0],
              r"Metric $g^{\theta,\theta}$"     : ggup_boz[js,:,0,1,1],
              r"Metric $g^{\theta,\zeta}$"      : ggup_boz[js,:,0,1,2],
              # r"Metric $g^{\zeta,\theta}$"      : ggup_boz[js,:,0,2,1],
              r"Metric $g^{\zeta,\zeta}$"       : ggup_boz[js,:,0,2,2],
             }

fig = plt.figure(figsize=(12,12))
fig.suptitle(r"$\rho$={:}".format(rho[js]))
ncol = 3
nrow = np.ceil(len(data_dicts)/ncol).astype(int)
for index, (key, value) in enumerate(data_dicts.items()):
    ax = fig.add_subplot(nrow,ncol,index+1)
    ax.plot(theta,value,".-")
    ax.set_xlabel(r"Boozer poloidal angle $\theta_B$")
    ax.set_ylabel(key)
    if value.min() < 0.0 and 0.0 < value.max():
        ax.axhline(0, lw=0.5, c="k")
fig.tight_layout()
plt.show()

# zeta_field_aligned = qq[js]*theta + alpha_fix
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(theta,zeta_field_aligned,"+-")
# ax.set_xlabel(r"Boozer poloidal angle $\theta_B$")
# ax.set_ylabel(r"Field-aligned Boozer toroidal angle $\zeta_B = qq(\rho)*\theta_B + \alpha_\mathrm{fix}$")
# plt.show()


# ### (iii) 3D plot of a field-aligned coordinates

# In[ ]:


X = rr*np.cos(ph)
Y = rr*np.sin(ph)
Z = zz

js = nss // 2

# flag_plot = "wireframe"
# flag_plot = "line"
flag_plot = "band"

fig = plt.figure(figsize=(22, 22))
ax = fig.add_subplot(111, projection='3d')
if flag_plot == "wireframe":
    ax.plot_wireframe(X[:,:,0], Y[:,:,0], Z[:,:,0], rstride=4, cstride=4, edgecolor='k', lw=0.5)
elif flag_plot == "line":
    ax.plot(X[js,:,0], Y[js,:,0], Z[js,:,0], "o-", c="r", lw=2)
elif flag_plot == "band":
    facecolors_value = bb[js-1:js+1,:,0]
    facecolors_label = "B [T]"
    norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
    colors = plt.cm.viridis(norm(facecolors_value))
    ax.plot_surface(X[js-1:js+1,:,0], Y[js-1:js+1,:,0], Z[js-1:js+1,:,0], facecolors=colors, rstride=1, cstride=1, edgecolor='k', lw=0.1)
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    mappable.set_array(facecolors_value)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)

# ax.view_init(elev=90, azim=0)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Magnetic Surface Reconstruction')
ax.set_zlim(-4,4)
# ax.set_xlim(-12,12)
# ax.set_ylim(-12,12)
plt.show()


# In[ ]:





# In[ ]:




