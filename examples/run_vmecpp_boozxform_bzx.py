#!/usr/bin/env python
# coding: utf-8

# ### Run VMEC++ code  
# VMEC ascii input file (input---)  
# -> VMEC MHD equilibrium, NetCDF output file (wout---.nc)

# In[ ]:


import vmecpp
vmecin = vmecpp.VmecInput.from_file("test_data/input.vmec_lhd")
vmecout = vmecpp.run(vmecin)
vmecout.wout.save("wout_lhd.nc")


# ### Run BOOZ_XFORM code  
# VMEC MHD equilibrium, NetCDF output file (wout---.nc)  
# -> Boozer coordinates, NetCDF output file (boozmn---.nc)  

# In[ ]:


import booz_xform as bx
b = bx.Booz_xform()
b.read_wout("wout_lhd.nc")
b.run()
b.write_boozmn("boozmn_lhd.nc")


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
fname_wout="wout_lhd.nc"
fname_boozmn="boozmn_lhd.nc"
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn, fname_wout, output_file="./metric_boozer.bin.dat")

nzeta = 128      # Only for check. toroidal grid number in [0 <= zeta <= 2*pi]
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, fname_boozmn="boozmn_lhd.nc", fname_wout="wout_lhd.nc", output_file="./metric_boozer_nzeta_finite.bin.dat")


# In[ ]:





# In[ ]:





# In[ ]:




