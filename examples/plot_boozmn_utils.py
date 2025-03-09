import numpy as np
import scipy as sp
import xarray as xr

def read_boozmn_nc(filename, flag_debug=False):
    """
    Read Boozer coordinate data from BOOZ_XFORM output NetCDF file boozmn---.nc

    Parameters:
    -----------
    filename : str
        File path for boozmn---.nc NetCDF file to read.
    flag_debug : bool, optional
        Flag to show debug information. Default is False.

    Return:
    --------
    tuple
        Tuple containing the followings:
        - nfp_b (int): The number of radial grid points in the equilibrium before the transformation to Boozer coordinates
        - ns_b (int): The number of identical field periods
        - aspect_b (float): Aspect ratio
        - rmax_b (float): -
        - rmin_b (float): -
        - betaxis_b (float): -
        - iota_b (numpy.ndarray): Rotational transform
        - pres_b (numpy.ndarray): -
        - beta_b (numpy.ndarray): -
        - phip_b (numpy.ndarray): -
        - phi_b (numpy.ndarray): - 
        - bvco_b (numpy.ndarray): Coefficient multiplying grad zeta_Boozer in the covariant representation of the magnetic field vector, often denoted G(psi) [Tesla * meter]
        - buco_b (numpy.ndarray): Coefficient multiplying grad theta_Boozer in the covariant representation of the magnetic field vector, often denoted I(psi) [Tesla * meter]
        - mboz_b (int): Maximum poloidal mode number m for which the Fourier amplitudes rmnc, bmnc etc are stored
        - nboz_b (int): Maximum toroidal mode number m for which the Fourier amplitudes rmnc, bmnc etc are stored
        - mnboz_b (int): The total number of (m,n) pairs for which Fourier amplitudes rmnc, bmnc etc are stored
        - jsize (int): Size of `jlist`
        - version (str): Version information
        - lasym_b (bool): Flag for stellerator-asymmetry. `False` if the configuration is stellerator-symmetric, `True` if the configuration is asymmetric.
        - ixn_b (numpy.ndarray): Poloidal mode numbers m for which the Fourier amplitudes rmnc, bmnc etc are stored
        - ixm_b (numpy.ndarray): Toroidal mode numbers n for which the Fourier amplitudes rmnc, bmnc etc are stored
        - jlist (numpy.ndarray): 1-based radial indices of the original vmec solution for which the transformation to Boozer coordinates was computed. 2 corresponds to the first half-grid point.
        - bmnc_b (numpy.ndarray): cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the magnetic field strength [Tesla]
        - rmnc_b (numpy.ndarray): cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the major radius R [meter]
        - zmns_b (numpy.ndarray): sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the height Z [meter]
        - pmns_b (numpy.ndarray): sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the angle difference zeta_VMEC - zeta_Boozer
        - gmnc_b (numpy.ndarray): cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the Boozer coordinate Jacobian (G + iota * I) / B^2 [meter/Tesla]
        - bmns_b (numpy.ndarray or None): sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the magnetic field strength [Tesla]
        - rmns_b (numpy.ndarray or None): sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the major radius R [meter]
        - zmnc_b (numpy.ndarray or None): cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the height Z [meter]
        - pmnc_b (numpy.ndarray or None): cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the angle difference zeta_VMEC - zeta_Boozer
        - gmns_b (numpy.ndarray or None): sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the Boozer coordinate Jacobian (G + iota * I) / B^2 [meter/Tesla]

    Example of usage:
    ----
    >>> filename = "boozmn_output.nc"  
    >>> boozmn_tuple = read_boozmn_nc(filename, flag_debug=True)  
    >>> nfp_b, ns_b, aspect_b, rmax_b, rmin_b, betaxis_b, \\  
    >>> iota_b_nu, pres_b_nu, beta_b_nu, phip_b_nu, phi_b_nu, bvco_b_nu, buco_b_nu, \\  
    >>> mboz_b, nboz_b, mnboz_b, jsize, version, lasym_b, \\  
    >>> ixn_b, ixm_b, jlist, \\  
    >>> bmnc_b, rmnc_b, zmns_b, pmns_b, gmnc_b, \\  
    >>> bmns_b, rmns_b, zmnc_b, pmnc_b, gmns_b = boozmn_tuple  
    """
    ds = xr.load_dataset(filename)
    
    # スカラー値の変数
    nfp_b = ds["nfp_b"].to_numpy()
    ns_b = ds["ns_b"].to_numpy()
    aspect_b = ds["aspect_b"].to_numpy()
    rmax_b = ds["rmax_b"].to_numpy()
    rmin_b = ds["rmin_b"].to_numpy()
    betaxis_b = ds["betaxis_b"].to_numpy()

    # 配列データ
    iota_b = ds["iota_b"].to_numpy()
    pres_b = ds["pres_b"].to_numpy()
    beta_b = ds["beta_b"].to_numpy()
    phip_b = ds["phip_b"].to_numpy()
    phi_b = ds["phi_b"].to_numpy()
    bvco_b = ds["bvco_b"].to_numpy()
    buco_b = ds["buco_b"].to_numpy()

    mboz_b = ds["mboz_b"].to_numpy()
    nboz_b = ds["nboz_b"].to_numpy()
    mnboz_b = ds["mnboz_b"].to_numpy()
    jsize = np.array(ds["jlist"].size, dtype=np.int32)

    # バージョン情報
    version = ds["version"].item().decode()

    # ステラレーター対称性フラグ
    lasym_b = bool(ds["lasym__logical__"])

    # インデックスデータ
    ixn_b = ds["ixn_b"].to_numpy()
    ixm_b = ds["ixm_b"].to_numpy()

    # jlist（トロイダル位置）
    jlist = ds["jlist"].to_numpy()

    # Boozer座標の物理量
    bmnc_b = ds["bmnc_b"].to_numpy()
    rmnc_b = ds["rmnc_b"].to_numpy()
    zmns_b = ds["zmns_b"].to_numpy()
    pmns_b = ds["pmns_b"].to_numpy()
    gmnc_b = ds["gmn_b"].to_numpy()

    # 非対称成分（lasym_b が True の場合）
    bmns_b = rmns_b = zmnc_b = pmnc_b = gmns_b = None
    if lasym_b:
        bmns_b = ds["bmns_b"].to_numpy()
        rmns_b = ds["rmns_b"].to_numpy()
        zmnc_b = ds["zmnc_b"].to_numpy()
        pmnc_b = ds["pmnc_b"].to_numpy()
        gmns_b = ds["gmns_b"].to_numpy()
        print("lasym_b =", lasym_b, "case is not yet checked.")

    # デバッグモード
    if flag_debug:
        print(ds.dims)
        print(ds.data_vars)

    # すべての設定した変数を返す
    return (nfp_b, ns_b, aspect_b, rmax_b, rmin_b, betaxis_b,
            iota_b, pres_b, beta_b, phip_b, phi_b, bvco_b, buco_b,
            mboz_b, nboz_b, mnboz_b, jsize, version, lasym_b,
            ixn_b, ixm_b, jlist,
            bmnc_b, rmnc_b, zmns_b, pmns_b, gmnc_b,
            bmns_b, rmns_b, zmnc_b, pmnc_b, gmns_b)




def read_wout_nc(filename, flag_debug=False):
    """
    NetCDF形式の VMEC 出力ファイル (wout_*.nc) から平衡データを読み込む。

    パラメータ:
    -----------
    filename : str
        読み込む NetCDF ファイルのパス。
    flag_debug : bool, オプション
        デバッグ情報を表示するかどうか。デフォルトは False。

    戻り値:
    --------
    tuple
        以下の要素を含むタプル:
        - B0_p (float): 磁場強度 (磁軸上の磁場 B0)。
        - Aminor_p (float): 小半径 (Aminor_p)。
        - Rmajor_p (float): 大半径 (Rmajor_p)。
        - volume_p (float): プラズマの体積 (volume_p)。

    例:
    ----
    >>> filename = "wout_li383_1.4m.nc"  
    >>> B0_p, Aminor_p, Rmajor_p, volume_p = read_wout_nc(filename, flag_debug=True)  
    >>> print("磁場強度 (B0):", B0_p)  
    >>> print("小半径 (Aminor_p):", Aminor_p)  
    >>> print("大半径 (Rmajor_p):", Rmajor_p)  
    >>> print("プラズマ体積 (volume_p):", volume_p)  
    """
    ds = xr.open_dataset(filename)

    B0_p = ds["b0"].to_numpy()
    Aminor_p = ds["Aminor_p"].to_numpy()
    Rmajor_p = ds["Rmajor_p"].to_numpy()
    volume_p = ds["volume_p"].to_numpy()
    
    # デバッグモード
    if flag_debug:
        print(ds.dims)
        print(ds.data_vars)
    
    return (B0_p, Aminor_p, Rmajor_p, volume_p)


def jmn2stz(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, theta_size=100, zeta_size=120):
    """ Fourier modes b[js,jmn] -> Scalar function b[js,jtheta,jzeta] """
    if bmnc_b is not None:
        jsize, mnboz = bmnc_b.shape # Radial grid points, total mode number
    elif bmns_b is not None:
        jsize, mnboz = bmns_b.shape

    ### Boozer coordinates (s,\theta_B,\zeta_B)
    hs = 1.0 / (ns_b - 1.0)
    compute_surfs = jlist - 2
    s_b = hs * (compute_surfs + 0.5) # Normalized flux label s
    theta_b = np.linspace(-np.pi, np.pi, theta_size) # Poloidal angle theta
    zeta_b = np.linspace(0, 2*np.pi/nfp_b, zeta_size)  # Toroidal angle zeta, with considering periodicity nfp_b

    ### Sum up over Fourier modes
    zeta, theta = np.meshgrid(zeta_b, theta_b, indexing="xy")
    # theta, zeta = np.meshgrid(theta_b, zeta_b, indexing="ij")
    m = ixm_b.reshape(mnboz,1,1) # Poloidal mode number
    n = ixn_b.reshape(mnboz,1,1) # Toroidal mode number
    angle = m * theta[np.newaxis,:,:] - n * zeta[np.newaxis,:,:]

    B = np.zeros([jsize,theta_size,zeta_size])
    if bmnc_b is not None:
        B += np.einsum("jm,mtz->jtz", bmnc_b, np.cos(angle))
    if bmns_b is not None:
        B += np.einsum("jm,mtz->jtz", bmns_b, np.sin(angle))

    return s_b, theta_b, zeta_b, B


def jmn2stz_at_js(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, js, theta_size=100, zeta_size=120):
    """ Fourier modes b[js,jmn] -> Scalar function b[jtheta,jzeta] at a given surface index js"""
    if bmnc_b is not None:
        jsize, mnboz = bmnc_b.shape # Radial grid points, total mode number
    elif bmns_b is not None:
        jsize, mnboz = bmns_b.shape

    ### Normalized flux label s in Boozer coordinates (s,\theta_B,\zeta_B)
    hs = 1.0 / (ns_b - 1.0)
    compute_surfs = jlist[js] - 2
    s_b = hs * (compute_surfs + 0.5)

    theta_b = np.linspace(-np.pi, np.pi, theta_size)
    zeta_b = np.linspace(0, 2*np.pi/nfp_b, zeta_size)  # `nfp_b` を考慮
    
    zeta, theta = np.meshgrid(zeta_b, theta_b, indexing="xy")
    # theta, zeta = np.meshgrid(theta_b, zeta_b, indexing="ij")
    m = ixm_b.reshape(mnboz,1,1) # Poloidal mode number
    n = ixn_b.reshape(mnboz,1,1) # Toroidal mode number
    angle = m * theta[np.newaxis,:,:] - n * zeta[np.newaxis,:,:]

    B = np.zeros([theta_size,zeta_size])
    if bmnc_b is not None:
        B += np.einsum("m,mtz->tz", bmnc_b[js,:], np.cos(angle))
    if bmns_b is not None:
        B += np.einsum("m,mtz->tz", bmns_b[js,:], np.sin(angle))

    return s_b, theta_b, zeta_b, B


def jmn2stz_at_zeta(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, zeta_b, theta_size=100):
    """ Fourier modes b[js,jmn] -> Scalar function b[js,jtheta] at a given Boozer toroidal angle zeta_b"""
    if bmnc_b is not None:
        jsize, mnboz = bmnc_b.shape # Radial grid points, total mode number
    elif bmns_b is not None:
        jsize, mnboz = bmns_b.shape

    ### Normalized flux label s in Boozer coordinates (s,\theta_B,\zeta_B)
    hs = 1.0 / (ns_b - 1.0)
    compute_surfs = jlist - 2
    s_b = hs * (compute_surfs + 0.5)

    theta_b = np.linspace(-np.pi, np.pi, theta_size)
    
    m = ixm_b.reshape(mnboz,1) # Poloidal mode number
    n = ixn_b.reshape(mnboz,1) # Toroidal mode number
    angle = m * theta_b[np.newaxis,:] - n * zeta_b

    B = np.zeros([jsize,theta_size])
    if bmnc_b is not None:
        B += np.einsum("jm,mt->jt", bmnc_b, np.cos(angle))
    if bmns_b is not None:
        B += np.einsum("jm,mt->jt", bmns_b, np.sin(angle))

    return s_b, theta_b, zeta_b, B


def jmn2stz_cartesian_grid(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, theta_size=100, zeta_size=120, flag_RZphi_output=False):
    # Major radius R
    s_b, theta_b, zeta_b, R = jmn2stz(rmnc_b, rmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, theta_size=theta_size, zeta_size=zeta_size)
    R = R + Rmajor_p
    # Height Z
    s_b, theta_b, zeta_b, Z = jmn2stz(zmnc_b, zmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, theta_size=theta_size, zeta_size=zeta_size)
    # Geometrical toroidal coordinate phi
    s_b, theta_b, zeta_b, phi_diff = jmn2stz(pmnc_b, pmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, theta_size=theta_size, zeta_size=zeta_size)
    phi = phi_diff + zeta_b[np.newaxis,np.newaxis,:]
    # Cartesian coordinates X,Y
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    if flag_RZphi_output:
        return R, Z, phi
    else:
        return X, Y, Z

def jmn2stz_cartesian_grid_at_js(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, js, theta_size=100, zeta_size=120, flag_RZphi_output=False):
    # Major radius R
    s_b, theta_b, zeta_b, R = jmn2stz_at_js(rmnc_b, rmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, js=js, theta_size=theta_size, zeta_size=zeta_size)
    R = R + Rmajor_p
    # Height Z
    s_b, theta_b, zeta_b, Z = jmn2stz_at_js(zmnc_b, zmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, js=js, theta_size=theta_size, zeta_size=zeta_size)
    # Geometrical toroidal coordinate phi
    s_b, theta_b, zeta_b, phi_diff = jmn2stz_at_js(pmnc_b, pmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, js=js, theta_size=theta_size, zeta_size=zeta_size)
    phi = phi_diff + zeta_b[np.newaxis,:]
    # Cartesian coordinates X,Y
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    if flag_RZphi_output:
        return R, Z, phi
    else:
        return X, Y, Z

def jmn2stz_cartesian_grid_at_zeta(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, Rmajor_p, zeta_b, theta_size=100, flag_RZphi_output=False):
    # Major radius R
    s_b, theta_b, zeta_b, R = jmn2stz_at_zeta(rmnc_b, rmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, zeta_b=zeta_b, theta_size=theta_size)
    R = R + Rmajor_p
    # Height Z
    s_b, theta_b, zeta_b, Z = jmn2stz_at_zeta(zmnc_b, zmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, zeta_b=zeta_b, theta_size=theta_size)
    # Geometrical toroidal coordinate phi
    s_b, theta_b, zeta_b, phi_diff = jmn2stz_at_zeta(pmnc_b, pmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, zeta_b=zeta_b, theta_size=theta_size)
    phi = phi_diff + zeta_b
    # Cartesian coordinates X,Y
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    if flag_RZphi_output:
        return R, Z, phi
    else:
        return X, Y, Z

def jmn2sta(bmnc_b, bmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, theta_size=100, alpha_size=120):
    """ Fourier modes b[js,jmn] -> Field-aligned scalar function b[js,jtheta,jalpha] """
    if bmnc_b is not None:
        jsize, mnboz = bmnc_b.shape # Radial grid points, total mode number
    elif bmns_b is not None:
        jsize, mnboz = bmns_b.shape

    ### Boozer coordinates (s,\theta_B,\zeta_B)
    hs = 1.0 / (ns_b - 1.0)
    compute_surfs = jlist - 2
    s_b = hs * (compute_surfs + 0.5) # Normalized flux label s
    theta_b = np.linspace(-np.pi, np.pi, theta_size) # Poloidal angle theta
    alpha_b = np.linspace(0, 2*np.pi/nfp_b, alpha_size)  # Toroidal angle zeta, with considering periodicity nfp_b

    ### Sum up over Fourier modes
    B = np.zeros((jsize,theta_size,alpha_size))
    zeta_b = np.empty((jsize,theta_size,alpha_size))
    for js in range(jsize):
        qq = 1.0/iota_b[1+js]
        for jt in range(theta_size):
            theta = theta_b[jt]
            for ja in range(alpha_size):
                zeta = qq * theta - alpha_b[ja]
                angle = ixm_b * theta - ixn_b * zeta # Shape: [mn]
                zeta_b[js,jt,ja] = zeta
                if bmnc_b is not None:
                    B[js,jt,ja] += np.dot(bmnc_b[js,:], np.cos(angle))
                if bmns_b is not None:
                    B[js,jt,ja] += np.dot(bmns_b[js,:], np.sin(angle))
    return s_b, theta_b, alpha_b, zeta_b, B


def jmn2sta_cartesian_grid(rmnc_b, zmns_b, pmns_b, rmns_b, zmnc_b, pmnc_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, Rmajor_p, theta_size=100, alpha_size=120, flag_RZphi_output=False):
    # Major radius R
    s_b, theta_b, alpha_b, zeta_b, R = jmn2sta(rmnc_b, rmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, theta_size, alpha_size)
    R = R + Rmajor_p
    # Height Z
    s_b, theta_b, alpha_b, zeta_b, Z = jmn2sta(zmnc_b, zmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, theta_size, alpha_size)
    # Geometrical toroidal coordinate phi
    s_b, theta_b, alpha_b, zeta_b, phi_diff = jmn2sta(pmnc_b, pmns_b, ixm_b, ixn_b, nfp_b, ns_b, jlist, iota_b, theta_size, alpha_size)
    # print(phi_diff.shape, theta_b.shape, zeta_b.shape)
    phi = phi_diff + zeta_b
    # Cartesian coordinates X,Y
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    if flag_RZphi_output:
        return R, Z, phi
    else:
        return X, Y, Z
