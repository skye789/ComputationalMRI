import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# load matlab file
mat = scipy.io.loadmat('digital_brain_phantom.mat')
label = mat['ph']['label'][0][0]
T1_map = mat['ph']['t1'][0][0]
T2_map = mat['ph']['t2'][0][0]
SD_map = mat['ph']['sd'][0][0] #M0
(nR,nC) = label.shape

# find segments
CSF_seg = (label==1)
GM_seg = (label==2)
WM_seg = (label==3)

# save plots of sementations
plt.imshow(CSF_seg,cmap='gray')
plt.title('brain phantom: label 1, CSF');
plt.savefig('temp1.png')

plt.imshow(GM_seg,cmap='gray')
plt.title('brain phantom: label 2, GM');
plt.savefig('temp2.png')

plt.imshow(WM_seg,cmap='gray')
plt.title('brain phantom: label 3, WM');
plt.savefig('temp3.png')

# get first index of segment and set value of that index
ind_CSF = np.argwhere(CSF_seg)
T1_CSF = T1_map[ind_CSF[0,0],ind_CSF[0,1]]
T2_CSF = T2_map[ind_CSF[0,0],ind_CSF[0,1]]
SD_CSF = SD_map[ind_CSF[0,0],ind_CSF[0,1]]
print ('T1 CSF = ' + str(T1_CSF) + ' ms')
print ('T2 CSF = ' + str(T2_CSF) + ' ms')
print ('SD CSF = ' + str(SD_CSF))

ind_GM = np.argwhere(GM_seg)
T1_GM = T1_map[ind_GM[0,0],ind_GM[0,1]]
T2_GM = T2_map[ind_GM[0,0],ind_GM[0,1]]
SD_GM = SD_map[ind_GM[0,0],ind_GM[0,1]]
print ('T1 GM = ' + str(T1_GM) + ' ms')
print ('T2 GM = ' + str(T2_GM) + ' ms')
print ('SD GM = ' + str(SD_GM))

# like in Matlab template, get all values of segment and average (same result)
T1_WM = np.mean(np.extract(WM_seg,T1_map))
T2_WM = np.mean(np.extract(WM_seg,T2_map))
SD_WM = np.mean(np.extract(WM_seg,SD_map))
print ('T1 WM = ' + str(T1_WM) + ' ms')
print ('T2 WM = ' + str(T2_WM) + ' ms')
print ('SD WM = ' + str(SD_WM))

