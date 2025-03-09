import numpy as np

def read_metric(fname_metric="./metric_boozer.bin.dat"):
    with open(fname_metric, 'rb') as f:
        header = np.fromfile(f,dtype='i4',count=1)
        nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b = np.fromfile(f,dtype='i4',count=7)
        Rax, Bax, aa, volume_p = np.fromfile(f,dtype='f8',count=4)
        asym_flg = np.fromfile(f,dtype='i4',count=1)
        alpha_fix = np.fromfile(f,dtype='f8',count=1)
        footer = np.fromfile(f,dtype='i4',count=1)
        if header != footer:
            print("Error 1 for reading file_name:", fname_metric)
    
        header = np.fromfile(f,dtype='i4',count=1)
        rho = np.fromfile(f,dtype='f8',count=nss)
        theta = np.fromfile(f,dtype='f8',count=ntht+1)
        zeta = np.fromfile(f,dtype='f8',count=nzeta+1)
        qq = np.fromfile(f,dtype='f8',count=nss)
        shat = np.fromfile(f,dtype='f8',count=nss)
        epst = np.fromfile(f,dtype='f8',count=nss)
        bb = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        rootg_boz = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        rootg_boz0 = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        ggup_boz = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)*3*3).reshape((3,3,nzeta+1,ntht+1,nss)).T
        dbb_drho = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        dbb_dtht = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        dbb_dzeta = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T 
        footer = np.fromfile(f,dtype='i4',count=1)
        if header != footer:
            print("Error 2 for reading file_name:", fname_metric)
    
        header = np.fromfile(f,dtype='i4',count=1)
        rr = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        zz = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T
        ph = np.fromfile(f,dtype='f8',count=nss*(ntht+1)*(nzeta+1)).reshape((nzeta+1,ntht+1,nss)).T 
        bbozc = np.fromfile(f,dtype='f8',count=mnboz_b*nss).reshape((nss,mnboz_b)).T
        ixn_b = np.fromfile(f,dtype='i4',count=mnboz_b)
        ixm_b = np.fromfile(f,dtype='i4',count=mnboz_b)
        footer = np.fromfile(f,dtype='i4',count=1)
        if header != footer:
            print("Error 3 for reading file_name:", fname_metric)
    
        if asym_flg == 1:
            header = np.fromfile(f,dtype='i4',count=1)
            bbozs = np.fromfile(f,dtype='f8',count=mnboz_b*nss).reshape((nss,mnboz_b)).T
            footer = np.fromfile(f,dtype='i4',count=1)
            if header != footer:
                print("Error 4 for reading file_name:", fname_metric)
        else:
            bbozs = None

    return (nfp_b, nss, ntht, nzeta, mnboz_b, mboz_b, nboz_b, Rax, Bax, aa, volume_p, asym_flg, alpha_fix,
            rho, theta, zeta, qq, shat, epst, bb, rootg_boz, rootg_boz0, ggup_boz,
            dbb_drho, dbb_dtht, dbb_dzeta,
            rr, zz, ph, bbozc, ixn_b, ixm_b,
            bbozs)