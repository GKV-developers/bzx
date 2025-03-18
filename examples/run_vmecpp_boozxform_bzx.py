#!/usr/bin/env python
# coding: utf-8

# ### Run VMEC++ code  
# VMEC ascii input file (input---)  
# -> VMEC MHD equilibrium, NetCDF output file (wout---.nc)

# In[ ]:


import vmecpp
vmecin = vmecpp.VmecInput.from_file("test_data/input.solovev")
vmecout = vmecpp.run(vmecin)
vmecout.wout.save("wout_solovev.nc")


# ### Run BOOZ_XFORM code  
# VMEC MHD equilibrium, NetCDF output file (wout---.nc)  
# -> Boozer coordinates, NetCDF output file (boozmn---.nc)  

# In[ ]:


import booz_xform as bx
b = bx.Booz_xform()
b.read_wout("wout_solovev.nc")
# # b.mboz = 20; b.nboz = 10 # If required to specify.
# print("Fourier poloidal mode number in Boozer coordinates", b.mboz)
# print("Fourier toroidal mode number in Boozer coordinates", b.nboz)
b.run()
b.write_boozmn("boozmn_solovev.nc")


# ### Run BZX code  
# Boozer coordinates, NetCDF output file (boozmn---.nc)  
# -> GKV coordinates, Fortran binary output file (metric_boozer.bin.dat)  

# In[ ]:


from bzx import bzx
Ntheta_gkv = 1   # N_tht value in GKV
nrho = 11        # radial grid number in [0 <= rho <= 1]
ntht = 384       # poloidal grid number in [-N_theta*pi < theta < N_theta*pi]
nzeta = 0        # nzeta==0 for output GKV field-aligned coordinates
alpha_fix = 0.0  # field-line label: alpha = zeta - q*theta NOT USED in 3d case (nzeta > 1)
fname_wout="wout_solovev.nc"
fname_boozmn="boozmn_solovev.nc"
output_file="./metric_boozer.bin.dat"
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn=fname_boozmn, fname_wout=fname_wout, output_file=output_file)
### When the "fname_wout" file is in ASCII format, "wout_txt" should be appropriately specified as written in "fname_wout", e.g.,
### bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn=fname_boozmn, fname_wout=fname_wout, output_file=output_file, wout_txt="solovev")

nzeta = 128      # Only for check. toroidal grid number in [0 <= zeta <= 2*pi]
output_file="./metric_boozer_nzeta_finite.bin.dat"
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn=fname_boozmn, fname_wout=fname_wout, output_file=output_file)


# In[ ]:





# In[ ]:




