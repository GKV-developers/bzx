#!/usr/bin/env python
# coding: utf-8

# ### Description
# Boozer coordinates $(s, \theta_B, \zeta_B)$ are a straight-field-line coordinate system,
# $$
# \boldsymbol{B} = \nabla \psi(s) \times \nabla \theta_B + \iota(\psi) \nabla \zeta_B \times \nabla \psi(s),
# $$
# and also expressed as
# $$
# \boldsymbol{B} = \beta(s,\theta_B,\zeta_B) \nabla \psi + I(s) \nabla \theta_B + G(s) \nabla \zeta_B.
# $$
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from plot_boozmn_utils import read_boozmn_nc, read_wout_nc
from plot_boozmn_utils import jmn2stz, jmn2stz_at_js, jmn2stz_at_zeta, jmn2stz_cartesian_grid, jmn2stz_cartesian_grid_at_js, jmn2stz_cartesian_grid_at_zeta
from plot_boozmn_utils import jmn2sta, jmn2sta_cartesian_grid

boozmn_ncfile = "boozmn_lhd.nc"
# help(read_boozmn_nc)
boozmn_tuple = read_boozmn_nc(boozmn_ncfile, flag_debug=True)
nfp_b, ns_b, aspect_b, rmax_b, rmin_b, betaxis_b, \
    iota_b, pres_b, beta_b, phip_b, phi_b, bvco_b, buco_b, \
    mboz_b, nboz_b, mnboz_b, jsize, version, lasym_b, \
    ixn_b, ixm_b, jlist, \
    bmnc_b, rmnc_b, zmns_b, pmns_b, gmnc_b, \
    bmns_b, rmns_b, zmnc_b, pmnc_b, gmns_b = boozmn_tuple

wout_ncfile = ("wout_lhd.nc")
B0_p, Aminor_p, Rmajor_p, volume_p = read_wout_nc(wout_ncfile)

print("Magnetic field strength at magnetic axis (B0_p):", B0_p)
print("Minor radius (Aminor_p):", Aminor_p)
print("Major radius (Rmajor_p):", Rmajor_p)
print("Plasma volume (volume_p):", volume_p)


# ## Fourier mode to real-space quantity
# ### (i) Radial profile

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(iota_b,".-")
ax.set_xlabel(r"Radial grid")
ax.set_ylabel(r"Rotational transform $\iota$")
plt.show()


# ### (ii) Magnetic field strength on magnetic surface

# In[ ]:


flag_plot_full_torus = True
if flag_plot_full_torus:
    nfp_b=1
print(nfp_b)

theta_size = 100
zeta_size = 120
s_b, theta_b, zeta_b, Bstz = jmn2stz(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, theta_size=theta_size, zeta_size=zeta_size)

js = jsize//4

fig = plt.figure()
ax = fig.add_subplot()
# quad=ax.contourf(zeta_b, theta_b, Bstz[js,:,:], 25)
s_b_at_js, theta_b, zeta_b, Bstz_at_js = jmn2stz_at_js(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, js=js, theta_size=theta_size, zeta_size=zeta_size)
quad=ax.contourf(zeta_b, theta_b, Bstz_at_js, 25)
ax.set_xlabel(r"Boozer toroidal angle $\zeta_B$")
ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
ax.set_title(r"|B| [T] on surface $s$={:.4f}".format(s_b[js]))
fig.colorbar(quad)
plt.show()


# ### (iii) Shape of magnetic flux surface

# In[ ]:


# X, Y, Z = jmn2stz_cartesian_grid(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, theta_size=theta_size, zeta_size=zeta_size)

js = jsize-1

facecolors_value = Bstz[js,:,:]
facecolors_label = "B [T]"

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
colors = plt.cm.viridis(norm(facecolors_value))
# ax.plot_surface(X[js,:,:], Y[js,:,:], Z[js,:,:], facecolors=colors, rstride=1, cstride=1, edgecolor='k', lw=0.1)
X_at_js, Y_at_js, Z_at_js = jmn2stz_cartesian_grid_at_js(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, js=js, theta_size=theta_size, zeta_size=zeta_size)
ax.plot_surface(X_at_js, Y_at_js, Z_at_js, facecolors=colors, rstride=1, cstride=1, edgecolor='k', lw=0.1)
mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(facecolors_value)
fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)
# ax.plot_wireframe(X[js,:,:], Y[js,:,:], Z[js,:,:], rstride=6, cstride=6, edgecolor='k', lw=0.5)
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
for jzeta in range(0,len(zeta_b),len(zeta_b)//4):
    facecolors_value = Bstz[:,:,jzeta]
    facecolors_label = "B [T]"
    norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
    colors = plt.cm.viridis(norm(facecolors_value))
    # ax.plot_surface(X[:,:,jzeta], Y[:,:,jzeta], Z[:,:,jzeta], facecolors=colors, rstride=20, cstride=1, edgecolor='k', lw=0.1)
    wzeta = zeta_b[jzeta]
    X_at_zeta, Y_at_zeta, Z_at_zeta = jmn2stz_cartesian_grid_at_zeta(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, wzeta, theta_size=theta_size)  
    ax.plot_surface(X_at_zeta, Y_at_zeta, Z_at_zeta, facecolors=colors, rstride=20, cstride=1, edgecolor='k', lw=0.1)
mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(facecolors_value)
fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)
js = len(jlist) - 1
X_at_js, Y_at_js, Z_at_js = jmn2stz_cartesian_grid_at_js(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, js=js, theta_size=theta_size, zeta_size=zeta_size)
ax.plot_wireframe(X_at_js, Y_at_js, Z_at_js, rstride=6, cstride=6, edgecolor='k', lw=0.5)
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


R, Z, phi = jmn2stz_cartesian_grid(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, theta_size=100, zeta_size=120, flag_RZphi_output=True)

theta_skip = 4
s_skip = 8
for izeta in range(0,len(zeta_b),len(zeta_b)//4):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(R[:,::theta_skip,izeta],Z[:,::theta_skip,izeta],c="k",lw=0.5)
    ax.plot(R[::s_skip,:,izeta].T,Z[::s_skip,:,izeta].T,c="k",lw=0.5)
    quad = ax.pcolormesh(R[:,:,izeta],Z[:,:,izeta],Bstz[:,:,izeta],vmin=Bstz.min(),vmax=Bstz.max())
    fig.colorbar(quad)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(R.min(),R.max())
    ax.set_ylim(Z.min(),Z.max())
    # ax.set_title(r"Poloidal cross section at $\zeta_B=${:.2f}".format(zeta_b[izeta]))
    ax.set_title(r"Poloidal projection of the cross section at $\zeta_B=${:.2f} [degree]".format(zeta_b[izeta]/np.pi*180))
    plt.show()


# ### (iv) Field-aligned plot

# In[ ]:


from time import time as timer
theta_size=50
alpha_size=30
t1 = timer()
s_b, theta_b, alpha_b, zeta_b_align, Bsta = jmn2sta(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, theta_size=theta_size, alpha_size=alpha_size)
t2 = timer(); print(t2-t1)

t1 = timer()
X_field_align, Y_field_align, Z_field_align = jmn2sta_cartesian_grid(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, Rmajor_p, theta_size=theta_size, alpha_size=alpha_size, flag_RZphi_output=False)
t2 = timer(); print(t2-t1)


# In[ ]:


js=jsize-1

facecolors_value = Bsta[js,:,:]
facecolors_label = "B [T]"

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(facecolors_value.min(), facecolors_value.max())
colors = plt.cm.viridis(norm(facecolors_value))
# ax.plot_surface(X_field_align[js,:,:], Y_field_align[js,:,:], Z_field_align[js,:,:], facecolors=colors, rstride=1, cstride=1, edgecolor='k', lw=0.1)
mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
mappable.set_array(facecolors_value)
fig.colorbar(mappable, ax=ax, shrink=0.6, label=facecolors_label)
ax.plot_wireframe(X_field_align[js,:,:], Y_field_align[js,:,:], Z_field_align[js,:,:], rstride=4, cstride=4, edgecolor='k', lw=0.5)
ax.plot(X_field_align[js,:,0], Y_field_align[js,:,0], Z_field_align[js,:,0], c="r", lw=2)
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





# In[ ]:




