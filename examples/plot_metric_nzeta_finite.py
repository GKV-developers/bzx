#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bzx import read_GKV_metric_file

fname_metric = "./metric_boozer_nzeta_finite.bin.dat"
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


assert nzeta > 0, "ERROR: plot_metric_nzeta_finite.py assumes nzeta>0."


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


# ### (ii) Magnetic field strength on magnetic surface

# In[ ]:


js = nss // 2

fig = plt.figure()
ax = fig.add_subplot()
# quad=ax.contourf(zeta_b, theta_b, Bstz[js,:,:], 25)
quad=ax.contourf(zeta, theta, bb[js,:,:], 25)
ax.set_xlabel(r"Boozer toroidal angle $\zeta_B$")
ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
ax.set_title(r"|B| [T] on surface $\rho$={:.4f}".format(rho[js]))
fig.colorbar(quad)
plt.show()


# ### (iii) Shape of magnetic flux surface

# In[ ]:


X = rr*np.cos(ph)
Y = rr*np.sin(ph)
Z = zz


js = nss-1

facecolors_value = bb[js,:,:]
facecolors_label = "B [T]"

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
colors = plt.cm.viridis(norm(facecolors_value))
ax.plot_surface(X[js,:,:], Y[js,:,:], Z[js,:,:], facecolors=colors, rstride=1, cstride=1, edgecolor='k', lw=0.1)
mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(facecolors_value)
fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)
# ax.plot_wireframe(X_at_js, Y_at_js, Z_at_js, rstride=6, cstride=6, edgecolor='k', lw=0.5)
# ax.view_init(elev=90, azim=0)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Magnetic Surface Reconstruction')
ax.set_zlim(-4,4)
# ax.set_xlim(-12,12)
# ax.set_ylim(-12,12)
# print(nfp_b)
plt.show()


# ### (iv) Slice at given Boozer angles

# In[ ]:


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
for jzeta in range(0,len(zeta)//nfp_b,(len(zeta)//nfp_b)//4):
    facecolors_value = bb[:,:,jzeta]
    facecolors_label = "B [T]"
    norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
    colors = plt.cm.viridis(norm(facecolors_value))
    ax.plot_surface(X[:,:,jzeta], Y[:,:,jzeta], Z[:,:,jzeta], facecolors=colors, rstride=20, cstride=1, edgecolor='k', lw=0.1)
    # wzeta = zeta_b[jzeta]
    # X_at_zeta, Y_at_zeta, Z_at_zeta = jmn2cartesian_grid_at_zeta(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, wzeta, theta_size=100)  
    # ax.plot_surface(X_at_zeta, Y_at_zeta, Z_at_zeta, facecolors=colors, rstride=20, cstride=1, edgecolor='k', lw=0.1)
mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(facecolors_value)
fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)
js = nss - 1
ax.plot_wireframe(X[js,:,:], Y[js,:,:], Z[js,:,:], rstride=6, cstride=6, edgecolor='k', lw=0.5)
# ax.view_init(elev=90, azim=0)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Magnetic Surface Reconstruction')
ax.set_zlim(-4,4)
# ax.set_xlim(-12,12)
# ax.set_ylim(-12,12)
# print(nfp_b)
plt.show()


# In[ ]:


theta_skip = 4
s_skip = 1
for jzeta in range(0,len(zeta)//nfp_b,(len(zeta)//nfp_b)//4):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(rr[:,::theta_skip,jzeta],zz[:,::theta_skip,jzeta],c="k",lw=0.5)
    ax.plot(rr[::s_skip,:,jzeta].T,zz[::s_skip,:,jzeta].T,c="k",lw=0.5)
    quad = ax.pcolormesh(rr[:,:,jzeta],zz[:,:,jzeta],bb[:,:,jzeta],vmin=bb.min(),vmax=bb.max())
    fig.colorbar(quad)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(rr.min(),rr.max())
    ax.set_ylim(zz.min(),zz.max())
    # ax.set_title(r"Poloidal cross section at $\zeta_B=${:.2f}".format(zeta_b[izeta]))
    ax.set_title(r"Poloidal projection of the cross section at $\zeta_B=${:.2f} [degree]".format(zeta[jzeta]/np.pi*180))
    plt.show()


# In[ ]:





# In[ ]:




