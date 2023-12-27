import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with h5py.File('Run_4.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print("List of datasets:\n", ls)
    NCapt = hdf.get('NCapt')[()]
    #print(NCapt)
    Prim = hdf.get('Prim')[()]

    posCaptT = NCapt['t']
    posCaptE = NCapt['E']
    posLocsX = NCapt['x'][:, 0]
    posLocsY = NCapt['x'][:, 1]
    posLocsZ = NCapt['x'][:, 2]
    posGamma = NCapt['Ngamma']
    posGammaE = NCapt['Egamma']
    posNprod = NCapt['Nprod']
    posVol = NCapt['vol']
    eventNos = NCapt['evt']
    IBDLocsX = Prim['x'][:, 0]
    IBDLocsY = Prim['x'][:, 1]
    IBDLocsZ = Prim['x'][:, 2]
    IBDVol = Prim['vol']
    IBDPID = Prim['PID']
    IBDevents = Prim['evt']

posPD = pd.DataFrame({'evt':eventNos, 'time':posCaptT, 'gammas':posGamma, 'Egammas':posGammaE, 'Pos x':posLocsX, 'Pos y':posLocsY, 'Pos z':posLocsZ, 'vol':posVol})
IBDPD = pd.DataFrame({'vol':IBDVol, 'evt':IBDevents, 'PID':IBDPID, 'x':IBDLocsX, 'y':IBDLocsZ, 'z':IBDLocsY})

posPD.set_index('evt')
IBDPD.set_index('evt')

IBDPD['z'] = IBDPD['z'] * -1

posPD = posPD[posPD.vol >= 0]
print(posPD.shape)
posPD = posPD[posPD.time < 0.15]
print(posPD.shape)
posPD = posPD[posPD.gammas == 2]
posPD = posPD[posPD.Egammas == 1.02199782]

IBDPD = IBDPD[IBDPD.vol >= 0]
IBDPD = IBDPD[IBDPD.PID == -11]

IBDPD = IBDPD[IBDPD.evt.isin(posPD.evt)]
posPD = posPD[posPD.evt.isin(IBDPD.evt)]

posPD = posPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

IBDx = IBDPD['x']
IBDy = IBDPD['y']
IBDz = IBDPD['z']
IBDevts = IBDPD['evt']

posPD = posPD.drop('Egammas', axis = 1)
posPD = posPD.drop('gammas', axis = 1)
posPD = posPD.drop('time', axis = 1)
posPD = posPD.drop('vol', axis = 1)

posPD = pd.concat([posPD, IBDx.rename('IBD x')], axis = 1)
posPD = pd.concat([posPD, IBDy.rename('IBD y')], axis = 1)
posPD = pd.concat([posPD, IBDz.rename('IBD z')], axis = 1)
posPD = pd.concat([posPD, IBDevts.rename('IBD evt')], axis = 1)

yRange = (posPD['IBD y'].max() - posPD['IBD y'].min()) / 2
posPD['IBD y'] = posPD['IBD y'] + (yRange - posPD['IBD y'].max())

posPD['X difference'] = posPD['Pos x'] - posPD['IBD x']
posPD['Y difference'] = posPD['Pos y'] - posPD['IBD y']
posPD['Z difference'] = posPD['Pos z'] - posPD['IBD z']

print(posPD.loc[posPD['evt'] != posPD['IBD evt']])
print(posPD.iloc[65890:65910])

posPD = posPD[posPD['IBD y'] < 773]
posPD = posPD[posPD['IBD y'] > -773]
#posPD = posPD[posPD['Pos y'] < 803]
#posPD = posPD[posPD['Pos y'] > -803]

posPD = posPD[posPD['IBD z'] < 557]
posPD = posPD[posPD['IBD z'] > -557]
#posPD = posPD[posPD['Pos z'] < 587]
#posPD = posPD[posPD['Pos z'] > -587]

posPD = posPD[posPD['IBD x'] < 992]
posPD = posPD[posPD['IBD x'] > -992]
#posPD = posPD[posPD['Pos x'] < 1022]
#posPD = posPD[posPD['Pos x'] > -1022]

IBDPD = IBDPD[IBDPD.evt.isin(posPD.evt)]
posPD = posPD[posPD.evt.isin(IBDPD.evt)]


posPD = posPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

posPD['X difference'] = posPD['Pos x'] - posPD['IBD x']
posPD['Y difference'] = posPD['Pos y'] - posPD['IBD y']
posPD['Z difference'] = posPD['Pos z'] - posPD['IBD z']

print(posPD.tail(20))

print("Range of IBD Y: ", posPD['IBD y'].min(), "to", posPD['IBD y'].max())
print("Range of Pos Y: ", posPD['Pos y'].min(), "to", posPD['Pos y'].max())

x_dif = posPD['X difference'].mean()
y_dif = posPD['Y difference'].mean()
z_dif = posPD['Z difference'].mean()
total_diff = np.sqrt(x_dif*x_dif + y_dif*y_dif + z_dif*z_dif)

print("Average X difference: ", x_dif)
print("Average Y difference: ", y_dif)
print("Average Z difference: ", z_dif)
print("Average total vector: ", total_diff)

plt.hist(posPD['IBD z'])
#plt.show()