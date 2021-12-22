import numpy as np

## GRAPPA R=2, kernel 2*3
def extract(acs, omit = None):
  nyacs, nxacs, nc = acs.shape 
  # calibration matrix

  # initial source matrix
  # 2*3 is the kernel size, nk
  # (nyacs-2)*(nxacs-2) is nb
  src = np.zeros(((nyacs-2)*(nxacs-2),nc*2*3), dtype=np.complex64)

  # initial target matrix
  targ = np.zeros(((nyacs-2)*(nxacs-2),nc), dtype=np.complex64)

  src_idx = 0
  for xi in range(1, nxacs-1):
    for yi in range(nyacs-2): # for循环所有target点(nb)
      if omit is not None:
        if yi%omit != 0:
          #print(acs[yi+1, xi])
          continue

      block = np.array([[acs[yi+dy, xi+dx] for dy in [0, 2]] for dx in [-1, 0, 1]]).flatten()
      src[src_idx] = block
      targ[src_idx] = acs[yi+1, xi]

      src_idx+=1

  return src, targ

def interp(acs, ws, omit):
  # 这里acs为zp_kspace,
  nyacs, nxacs, nc = acs.shape 
  # calibration matrix

  # initial source matrix
  src = np.zeros(((nyacs-2)*(nxacs-2),nc*2*3), dtype=np.complex64)#2*3 is the kernel size

  # initial target matrix
  targ = np.zeros(((nyacs-2)*(nxacs-2),nc), dtype=np.complex64)

  src_idx = 0
  interpolated = np.array(acs)
  for xi in range(1, nxacs-1):
    for yi in range(nyacs-2):
      if yi%omit != 0:
          #print(acs[yi+1, xi])
          continue

      block = np.array([[acs[yi+dy, xi+dx] for dy in [0, 2]] for dx in [-1, 0, 1]]).flatten() #2×3×8
      #src[src_idx] = block
      #print(block.shape)
      interpolated[yi+1, xi] = np.dot(block, ws) #ws:
      targ[src_idx] = acs[yi+1, xi]

      src_idx+=1

  return interpolated, acs, targ

def grappaR2K2x3(kdata, acs, flag_acs = False):
  '''''
  Input:
    kdata: nx, ny, nc, every other line to be zero
    acs: nxacs, nyacs, nc
    flag_acs: whether use the acs in the reconstructed kspace or re-interpolate the center part of kspace
  Output:
    interpolated: nx, ny, nc
    zp_kdata: nx+1, ny+2, nc


  GRAPPA reconstruction of 2D image with uniform 2-fold acceleration along
  the ky dimension using a 2x3 kernel

  Keyword arguments:
  kdata -- accelerated acquisition kspace data, with every other lines to be zeros, numpy.array of shape [ny, nx, nc]
  acs -- autocalibration signal(Fully sampled part of the kspace) [nyacs, nxacs, nc]
  flag_acs -- if include the autocalibration data in the final reconstruction

  kernel location:
  111
  000
  111

  target:
  000
  010
  000

  Based on Ricardo Otazo's MATLAB implementation

  Zhengnan Huang Mon Oct  5 18:42:01 EDT 2020
  '''

  ny, nx, nc = kdata.shape
  nyacs, nxacs, _ = acs.shape

  # get source and target in acs region, to calculate weight
  src, targ = extract(acs)
  ws=np.dot(np.linalg.pinv(src), targ)#different from original implementation
  
  #Zero pad the kspace matrix to make sure every skipped line been filed
  # zero pad a 'U' structure
  #from
  #111
  #000
  #111
  #000
  #to 
  #z111z
  #z000z
  #z111z
  #z000z
  #zzzzz
  zp_kdata = np.zeros((ny+1, nx+2, nc), np.complex64)
  zp_kdata[:ny, 1:nx+1, :] = kdata

  #interpolation
  interpolated, _, _ = interp(zp_kdata, ws, omit=2)
  interpolated = interpolated[:ny, 1:nx+1]
  #print("The second number should be 0", temp.shape, temp.sum())
  
  if flag_acs:
    interpolated[int(ny/2-nyacs/2):int(ny/2+nyacs/2)] = acs
  #interpolated
  return interpolated, zp_kdata

#interpolated, zp_kdata = grappaR2K2x3(uskspace, acs_data, flag_acs=True)
#print(interpolated.shape)

