import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open HDF5 file and extract datasets and tables
with h5py.File('Annihilation.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print("List of datasets:\n", ls)

    NCapt = hdf.get('NCapt')[()]
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

# Two dataframes to contain the two tables. Note that y and z are flipped for IBDs
posPD = pd.DataFrame({'evt':eventNos, 'time':posCaptT, 'gammas':posGamma, 'Egammas':posGammaE, 'Pos x':posLocsX, 'Pos y':posLocsY, 'Pos z':posLocsZ, 'vol':posVol})
IBDPD = pd.DataFrame({'vol':IBDVol, 'evt':IBDevents, 'PID':IBDPID, 'x':IBDLocsX, 'y':IBDLocsZ, 'z':IBDLocsY})

# Extract reactor core position
reactorPos = np.array([IBDPD.at[0, 'x'], IBDPD.at[0, 'y'], IBDPD.at[0, 'z']])
print("Original reactor position (mm): \nx =", reactorPos[0], "\ny =", reactorPos[1], "\nz = ", reactorPos[2])

# Setting index for ease of use
posPD.set_index('evt')
IBDPD.set_index('evt')

# Primaries table has z backwards for some reason
IBDPD['z'] = IBDPD['z'] * -1
reactorPos[2] = reactorPos[2] * -1

# Dropping events out of segments
posPD = posPD[posPD.vol >= 0]
posPD = posPD[posPD.vol <= 153]
IBDPD = IBDPD[IBDPD.vol >= 0]
IBDPD = IBDPD[IBDPD.vol <= 153]

# Dropping events that aren't proper positron annihilations
posPD = posPD[posPD.time < 0.14]
posPD = posPD[posPD.gammas == 2]
posPD = posPD[posPD.Egammas == 1.02199782]
posPD = posPD.drop_duplicates(subset=['evt'])

# Dropping all rows in IBD dataframe that aren't positrons
IBDPD = IBDPD[IBDPD.PID == -11]

# Aligning both tables to drop the same events
IBDPD = IBDPD[IBDPD.evt.isin(posPD.evt)]
posPD = posPD[posPD.evt.isin(IBDPD.evt)]

# Make counting and combining easier by resetting indices
posPD = posPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

# Extracting columns to combine into one dataframe
IBDx = IBDPD['x']
IBDy = IBDPD['y']
IBDz = IBDPD['z']
IBDevts = IBDPD['evt']

# Don't need these columns anymore
posPD = posPD.drop('Egammas', axis = 1)
posPD = posPD.drop('gammas', axis = 1)
posPD = posPD.drop('time', axis = 1)
posPD = posPD.drop('vol', axis = 1)

# Combining two dataframes
posPD = pd.concat([posPD, IBDx.rename('IBD x')], axis = 1)
posPD = pd.concat([posPD, IBDy.rename('IBD y')], axis = 1)
posPD = pd.concat([posPD, IBDz.rename('IBD z')], axis = 1)
posPD = pd.concat([posPD, IBDevts.rename('IBD evt')], axis = 1)

# Y is offset so fixing that by centering the range around 0 
posPD['IBD y'] = posPD['IBD y'] + 743.1281975

# Adding columns for each differece
posPD['X difference'] = posPD['Pos x'] - posPD['IBD x']
posPD['Y difference'] = posPD['Pos y'] - posPD['IBD y']
posPD['Z difference'] = posPD['Pos z'] - posPD['IBD z']

# Checking if there are still misalignments
print("If there are any misaligned events, it will print now:\n", posPD.loc[posPD['evt'] != posPD['IBD evt']])
posPD.drop('IBD evt', axis=1)

# Cutting a few mm (30) inside the active detector. Only applied to IBDs
posPD = posPD[posPD['IBD y'] < 773]
posPD = posPD[posPD['IBD y'] > -773]
# posPD = posPD[posPD['Pos y'] < 803]
# posPD = posPD[posPD['Pos y'] > -803]

posPD = posPD[posPD['IBD z'] < 557]
posPD = posPD[posPD['IBD z'] > -557]
# posPD = posPD[posPD['Pos z'] < 587]
# posPD = posPD[posPD['Pos z'] > -587]

posPD = posPD[posPD['IBD x'] < 1019]
posPD = posPD[posPD['IBD x'] > -1019]
# posPD = posPD[posPD['Pos x'] < 1022]
# posPD = posPD[posPD['Pos x'] > -1022]

# Aligning tables again just for safety
IBDPD = IBDPD[IBDPD.evt.isin(posPD.evt)]
posPD = posPD[posPD.evt.isin(IBDPD.evt)]

# Make counting easier by resetting indices
posPD = posPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

# Adding difference columns
posPD['X difference'] = posPD['Pos x'] - posPD['IBD x']
posPD['Y difference'] = posPD['Pos y'] - posPD['IBD y']
posPD['Z difference'] = posPD['Pos z'] - posPD['IBD z'] 

# Getting vector from IBD to reactor core for each event
reactorVectorx = reactorPos[0] - posPD['IBD x']
reactorVectory = reactorPos[1] - posPD['IBD y']
reactorVectorz = reactorPos[2] - posPD['IBD z']

# Getting projected vector for positron annihilation along IBD to reactor core
projectedVector = (reactorVectorx * posPD['X difference'] + reactorVectory * posPD['Y difference'] + reactorVectorz * posPD['Z difference']) / np.sqrt(reactorVectorx * reactorVectorx + reactorVectory * reactorVectory + reactorVectorz * reactorVectorz)

# Printing preview of dataframe to terminal
print(posPD.head(20))

# Gathering the average difference in each direction and the average total difference
x_dif = posPD['X difference'].mean()
y_dif = posPD['Y difference'].mean()
z_dif = posPD['Z difference'].mean()
total_diff = np.sqrt(x_dif*x_dif + y_dif*y_dif + z_dif*z_dif)
avgProjVector = projectedVector.mean()

# Printing to terminal
print("Average X difference: ", x_dif)
print("Average Y difference: ", y_dif)
print("Average Z difference: ", z_dif)
print("Average total vector: ", total_diff)
print("Average total vector projected along the vector from the IBD location towards the reactor core: ", avgProjVector)

# Plots for verification
plt.figure()
plt.hist(posPD['Pos x'])
plt.xlabel("X axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Positron Annihilations vs X")

plt.figure()
plt.hist(posPD['Pos y'])
plt.xlabel("Y axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Positron Annihilations vs Y")

plt.figure()
plt.hist(posPD['Pos z'])
plt.xlabel("Z axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Positron Annihilations vs Z")

plt.figure()
plt.hist(posPD['X difference'], bins=100)
plt.xlabel("X axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Positron locations in X")

plt.figure()
plt.hist(posPD['Y difference'], bins=100)
plt.xlabel("Y axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Positron locations in Y")

plt.figure()
plt.hist(posPD['Z difference'], bins=100)
plt.xlabel("Z axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Positron locations in Z")

plt.show()