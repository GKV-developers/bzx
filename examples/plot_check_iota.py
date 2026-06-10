# %%
import numpy as np
import matplotlib.pyplot as plt
from plot_boozmn_utils import read_boozmn_nc, read_wout_nc
from plot_boozmn_utils import jmn2stz, jmn2stz_at_js, jmn2stz_at_zeta, jmn2stz_cartesian_grid, jmn2stz_cartesian_grid_at_js, jmn2stz_cartesian_grid_at_zeta
from plot_boozmn_utils import jmn2sta, jmn2sta_cartesian_grid

boozmn_ncfile = "boozmn_w7x.nc"
# help(read_boozmn_nc)
boozmn_tuple = read_boozmn_nc(boozmn_ncfile, flag_debug=True)
nfp_b, ns_b, aspect_b, rmax_b, rmin_b, betaxis_b, \
    iota_b, pres_b, beta_b, phip_b, phi_b, bvco_b, buco_b, \
    mboz_b, nboz_b, mnboz_b, jsize, version, lasym_b, \
    ixn_b, ixm_b, jlist, \
    bmnc_b, rmnc_b, zmns_b, pmns_b, gmnc_b, \
    bmns_b, rmns_b, zmnc_b, pmnc_b, gmns_b = boozmn_tuple

wout_ncfile = ("wout_w7x.nc")
B0_p, Aminor_p, Rmajor_p, volume_p = read_wout_nc(wout_ncfile)

print("Magnetic field strength at magnetic axis (B0_p):", B0_p)
print("Minor radius (Aminor_p):", Aminor_p)
print("Major radius (Rmajor_p):", Rmajor_p)
print("Plasma volume (volume_p):", volume_p)

# %%
import xarray as xr
ds = xr.open_dataset("wout_w7x.nc")
print("iota_b.shape:", iota_b.shape)
print("iotas.shape:", ds.iotas.shape)
print("iotaf.shape:", ds.iotaf.shape)

rho_wout = np.sqrt(ds.phi/ds.phi[-1]).to_numpy()
phi_half = np.array(0.5*(ds.phi[1:] + ds.phi[:-1]))
rho_wout_half = np.sqrt(phi_half / float(ds.phi[-1]))
print(rho_wout.shape, rho_wout_half.shape)
rho_boozmn = np.sqrt(phi_b/phi_b[-1])
phi_b_half = np.array(0.5*(phi_b[1:] + phi_b[:-1]))
rho_boozmn_half = np.sqrt(phi_b_half / phi_b[-1])

from bzx import read_GKV_metric_file
fname_metric = "./metric_boozer.bin.dat"
nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b, Rax, Bax, aa, volume_p, asym_flg, alpha_fix, \
    rho, theta, zeta, qq, shat, epst, bb, rootg_boz, rootg_boz0, ggup_boz, \
    dbb_drho, dbb_dtht, dbb_dzeta, \
    rr, zz, ph, bbozc, ixn_b, ixm_b, \
    bbozs = read_GKV_metric_file(fname_metric)
print(qq.shape)
print(float(ds.phips[-1]))
print(float(ds.phipf[-1]))

plt.plot(rho,1/qq,"s-",label="1/qq (BZX output: metric_boozer.bin.dat)")
plt.plot(rho_boozmn,iota_b,".",label="iota_b plotted on full rho")
plt.plot(rho_boozmn_half,iota_b[1:],".",label="iota_b plotted on half rho")
plt.plot(rho_wout,ds.iotas,"+",label="iotas plotted on full rho")
plt.plot(rho_wout_half,ds.iotas[1:],"+",label="iotas plotted on half rho")
plt.plot(rho_wout,ds.iotaf,"x",label="iotaf plotted on full rho")
plt.xlim(-0.01,0.2)
plt.ylim(None,0.9)
plt.xlabel(r"$\rho = \sqrt{\Phi / \Phi_a}$")
# plt.xticks(np.arange(0,21,5))
plt.grid()
plt.legend()
plt.show()

plt.plot(rho,1/qq,"s-",label="1/qq (BZX output: metric_boozer.bin.dat)")
plt.plot(rho_boozmn,iota_b,".",label="iota_b plotted on full rho")
plt.plot(rho_boozmn_half,iota_b[1:],".",label="iota_b plotted on half rho")
plt.plot(rho_wout,ds.iotas,"+",label="iotas plotted on full rho")
plt.plot(rho_wout_half,ds.iotas[1:],"+",label="iotas plotted on half rho")
plt.plot(rho_wout,ds.iotaf,"x",label="iotaf plotted on full rho")
plt.xlim(0.55,0.7)
plt.ylim(0.88,0.9)
plt.xlabel(r"$\rho = \sqrt{\Phi / \Phi_a}$")
plt.grid()
plt.legend()
plt.show()


plt.plot(rho,1/qq,"s-",label="1/qq (BZX output: metric_boozer.bin.dat)")
plt.plot(rho_boozmn,iota_b,".",label="iota_b plotted on full rho")
plt.plot(rho_boozmn_half,iota_b[1:],".",label="iota_b plotted on half rho")
plt.plot(rho_wout,ds.iotas,"+",label="iotas plotted on full rho")
plt.plot(rho_wout_half,ds.iotas[1:],"+",label="iotas plotted on half rho")
plt.plot(rho_wout,ds.iotaf,"x",label="iotaf plotted on full rho")
plt.xlim(0.9,1.01)
plt.ylim(0.925,0.95)
plt.xlabel(r"$\rho = \sqrt{\Phi / \Phi_a}$")
plt.grid()
plt.legend()
plt.show()
# %%
