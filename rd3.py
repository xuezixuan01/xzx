import shutil
import os
from bruker.api.topspin import Topspin
from bruker.data.nmr  import *
import math
from bruker.publish.plot import *
import matplotlib.pyplot as plt
import numpy as np
from bruker.data.direct_io import NMRDataSetDirect
###

# jcoup = [59.25, 3.31, 1.37, 1.16, 0.95, 0.74]
# abund = [0.00555, 0.0247, 0.0016, 0.0047, 0.0047, 0.0016]
# shift = [-0.00155, -0.00013, -0.0002, -0.0002, -0.0002, -0.0002]
# elif type == 'TSP':
# jcoup = [59.25, 3.31, 1.37, 1.16, 0.95, 0.74]
# abund = [0.00555, 0.0247, 0.0016 * l1, 0.0047 * l1, 0.0047 * l1, 0.0016 * l1]
# shift = [-0.00155, -0.00013, -0.0002, -0.0002, -0.0002, -0.0002]

def reference_type(a,type,fre,cent,i,l2,sf):
    l1=2/3*0
    jcoup = []
    abund = []
    shift =[]
    if type=='single':
        return a
    elif type=='TMS':
        jcoup=[3.31,1.37,1.16,0.95,0.74]
        abund=[0.0247,0.0016,0.0047,0.0047,0.0016]
        shift=[-0.00013,-0.0002,-0.0002,-0.0002,-0.0002]
    elif type=='TSP':
        jcoup=[3.31,1.37,1.16,0.95,0.74]
        abund=[0.0247,0.0016*l1,0.0047*l1,0.0047*l1,0.0016*l1]
        shift = [-0.00013, -0.0002, -0.0002, -0.0002, -0.0002]
    for j in range (0, len(jcoup)):
        jshift1=round(cent-jcoup[j]*fre*l2+shift[j]*sf*fre*l2)
        jshift2 = round(cent + jcoup[j]*fre*l2+shift[j]*sf*fre*l2)
        ji = i *abund[j]
        a[jshift1]=ji
        a[jshift2] = ji
    return a

top = Topspin()
dp = top.getDataProvider()
dpa=dp.getCurrentDatasetIdentifier()
###################################

top.getDisplay().closeAllWindows()
#
# Process data sets required for the examples working
#
#old_dataset_path=top.getInstallationDirectory() + '/examdata/Reference_deconvolution/1/pdata/1'
old_dataset_path=dpa
proton = dp.getNMRData(old_dataset_path)
top.getDisplay().show(proton)
#new_dataset_path = top.getInstallationDirectory() + '/examdata/Reference_deconvolution/2/pdata/1'

proton.launch('efp')
# create new procno
proton.launch('wrp 999 y')
proton.launch('ft')
proton.launch('apk')
proton.launch('abs')
#proton1 = dp.getNMRData(new_dataset_path)
proton2 = dp.getNMRData(proton.getIdentifier(), directIO=True, procno=999)

###########################################
###########################################
t=proton.getSpecDataPoints()
#f=proton.getRawDataPoints()
usep1=proton.getPar('1 USERP1')
sf=float(proton.getPar('SF'))
ulist=[]
for ui in range(0,len(usep1)):
    if usep1[ui]==",":
        ulist.append(ui)

left_point=int(usep1[0:ulist[0]])
right_point=int(usep1[ulist[0]+1:ulist[1]])
center_point=int(usep1[ulist[1]+1:ulist[2]])
hzlb=float(usep1[ulist[2]+1:ulist[3]])
hzgb=float(usep1[ulist[3]+1:ulist[4]])
type=usep1[ulist[4]+1:len(usep1)]
#top.executeCommand('1 USERP1')
hz_total = proton.basicPars["status SW_p"]
ppmrange = proton.basicPars["status OFFSET"]
orgspec=t['dataPoints']
resarea=t['dataPoints'].copy()
intRegions = proton.getIntegrationRegions()



for i in range(0,len(resarea)):
    if i<left_point:
        resarea[i]=0
    elif i>right_point:
        resarea[i]=0
##########################################
point_num=len(orgspec)


if abs(hzlb)>0.01:
    lb=hzlb/hz_total
    te = 1 / (math.pi * lb)
if abs(hzgb)>0.01:
    gb=hzgb/hz_total
    tg=2*(math.log(2))**0.5/gb/math.pi
aj=100000

ref_fid=[]
plotp=[]
prexx=[]
for i in range(0,len(resarea)):
    prexx.append(0)
    plotp.append(i)

for p in range(0,len(prexx)):
    if abs(p-center_point)>0.1:
        prexx[p]=0
    else:
        prexx[p]=100000
###################
prexx=reference_type(prexx,type,len(prexx)/ppmrange,center_point,100000,ppmrange/hz_total,sf)
fpxx=np.fft.ifft(prexx)
# costime=center_point/len(prexx)*2*math.pi
# for i in range(0,len(fpxx)):
#     fpxx[i]=1000*(np.cos(costime*i)+1j*np.sin(costime*i))
x=[]
lineshape=0
if abs(hzgb)>=0.01:
    if abs(hzlb)>=0.01:
        lineshape=1
    else:
        lineshape=2
else:
    if abs(hzlb)>=0.01:
        lineshape=3
if lineshape==1:
    for t in range(0,len(resarea)):
        x.append(t)
        tf=t
        ref_fid.append(fpxx[t]*(math.e**(-tf/te))*math.e**(-(tf*tf/(tg*tg))))
elif lineshape==2:
    for t in range(0,len(resarea)):
        x.append(t)
        tf=t
        ref_fid.append(fpxx[t]*math.e**(-(tf*tf/(tg*tg))))
elif lineshape==3:
    for t in range(0,len(resarea)):
        x.append(t)
        tf=t
        ref_fid.append(fpxx[t]*(math.e**(-tf/te)))
#########################################
act_fid=np.fft.ifft(resarea)
c=ref_fid/act_fid
############################
org_fid=np.fft.ifft(orgspec)
result_fid=org_fid*c
newspec1=np.fft.fft(np.real(result_fid))
newspec2=np.fft.fft(np.imag(result_fid))
newspec=newspec1+1j*newspec2
newspecr=np.real(newspec)

proton2.setDataVector(PROCDATA, newspecr)
#proton2.setDataVector(RAWDATA, newfid)
proton2.launch('abs')
# show the data sets on the TopSpin display
top.getDisplay().show(proton2,'ture')
top.getDisplay().arrangeWindowsVertical()



