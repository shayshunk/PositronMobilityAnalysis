import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open HDF5 file and extract datasets and tables
with h5py.File('NCapt.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print("List of datasets:\n", ls)

    NCapt = hdf.get('NCapt')[()]
    Prim = hdf.get('Prim')[()]

    NLocsX = NCapt['x'][:, 0]
    NLocsY = NCapt['x'][:, 1]
    NLocsZ = NCapt['x'][:, 2]
    NVol = NCapt['vol']
    eventNos = NCapt['evt']
    IBDLocsX = Prim['x'][:, 0]
    IBDLocsY = Prim['x'][:, 1]
    IBDLocsZ = Prim['x'][:, 2]
    IBDVol = Prim['vol']
    IBDPID = Prim['PID']
    IBDevents = Prim['evt']

# Two dataframes to contain the two tables. Note that y and z are flipped for IBDs
NCaptPD = pd.DataFrame({'evt':eventNos, 'N x':NLocsX, 'N y':NLocsY, 'N z':NLocsZ, 'vol':NVol})
IBDPD = pd.DataFrame({'vol':IBDVol, 'evt':IBDevents, 'PID':IBDPID, 'x':IBDLocsX, 'y':IBDLocsZ, 'z':IBDLocsY})

# Extract reactor core position
reactorPos = np.array([IBDPD.at[0, 'x'], IBDPD.at[0, 'y'], IBDPD.at[0, 'z']])
print("Original reactor position (mm): \nx =", reactorPos[0], "\ny =", reactorPos[1], "\nz = ", reactorPos[2])

# Setting index for ease of use
NCaptPD.set_index('evt')
IBDPD.set_index('evt')

# Primaries table has z backwards for some reason
IBDPD['z'] = IBDPD['z'] * -1
reactorPos[2] = reactorPos[2] * -1

# Dropping events out of segments
NCaptPD = NCaptPD[NCaptPD.vol >= 0]
NCaptPD = NCaptPD[NCaptPD.vol <= 153]
IBDPD = IBDPD[IBDPD.vol >= 0]
IBDPD = IBDPD[IBDPD.vol <= 153]

# Dropping events that aren't proper neutron captures
NCaptPD = NCaptPD.drop_duplicates(subset=['evt'])

# Dropping all rows in IBD dataframe that aren't neutron events
IBDPD = IBDPD[IBDPD.PID == 2112]

# Aligning both tables to drop the same events
IBDPD = IBDPD[IBDPD.evt.isin(NCaptPD.evt)]
NCaptPD = NCaptPD[NCaptPD.evt.isin(IBDPD.evt)]

# Make counting and combining easier by resetting indices
NCaptPD = NCaptPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

# Extracting columns to combine into one dataframe
IBDx = IBDPD['x']
IBDy = IBDPD['y']
IBDz = IBDPD['z']
IBDevts = IBDPD['evt']

# Don't need these columns anymore
NCaptPD = NCaptPD.drop('vol', axis = 1)

# Combining two dataframes
NCaptPD = pd.concat([NCaptPD, IBDx.rename('IBD x')], axis = 1)
NCaptPD = pd.concat([NCaptPD, IBDy.rename('IBD y')], axis = 1)
NCaptPD = pd.concat([NCaptPD, IBDz.rename('IBD z')], axis = 1)
NCaptPD = pd.concat([NCaptPD, IBDevts.rename('IBD evt')], axis = 1)

# Y is offset so fixing that by centering the range around 0 
NCaptPD['IBD y'] = NCaptPD['IBD y'] + 743.1281975

# Adding columns for each differece
NCaptPD['X difference'] = NCaptPD['N x'] - NCaptPD['IBD x']
NCaptPD['Y difference'] = NCaptPD['N y'] - NCaptPD['IBD y']
NCaptPD['Z difference'] = NCaptPD['N z'] - NCaptPD['IBD z']

# Checking if there are still misalignments
print("If there are any misaligned events, it will print now:\n", NCaptPD.loc[NCaptPD['evt'] != NCaptPD['IBD evt']])
NCaptPD.drop('IBD evt', axis=1)

# Cutting a few mm (30) inside the active detector. Only applied to IBDs
NCaptPD = NCaptPD[NCaptPD['IBD y'] < 773]
NCaptPD = NCaptPD[NCaptPD['IBD y'] > -773]
# NCaptPD = NCaptPD[NCaptPD['Pos y'] < 803]
# NCaptPD = NCaptPD[NCaptPD['Pos y'] > -803]

NCaptPD = NCaptPD[NCaptPD['IBD z'] < 557]
NCaptPD = NCaptPD[NCaptPD['IBD z'] > -557]
# NCaptPD = NCaptPD[NCaptPD['Pos z'] < 587]
# NCaptPD = NCaptPD[NCaptPD['Pos z'] > -587]

NCaptPD = NCaptPD[NCaptPD['IBD x'] < 1019]
NCaptPD = NCaptPD[NCaptPD['IBD x'] > -1019]
# NCaptPD = NCaptPD[NCaptPD['Pos x'] < 1022]
# NCaptPD = NCaptPD[NCaptPD['Pos x'] > -1022]

# Aligning tables again just for safety
IBDPD = IBDPD[IBDPD.evt.isin(NCaptPD.evt)]
NCaptPD = NCaptPD[NCaptPD.evt.isin(IBDPD.evt)]

# Make counting easier by resetting indices
NCaptPD = NCaptPD.reset_index(drop=True)
IBDPD = IBDPD.reset_index(drop=True)

# Adding difference columns
NCaptPD['X difference'] = NCaptPD['N x'] - NCaptPD['IBD x']
NCaptPD['Y difference'] = NCaptPD['N y'] - NCaptPD['IBD y']
NCaptPD['Z difference'] = NCaptPD['N z'] - NCaptPD['IBD z'] 
NCaptPD['Total difference'] = np.sqrt(NCaptPD['X difference']*NCaptPD['X difference'] + NCaptPD['Y difference']*NCaptPD['Y difference'] + NCaptPD['Z difference']*NCaptPD['Z difference'])

# Getting vector from IBD to reactor core for each event
reactorVectorx = reactorPos[0] - NCaptPD['IBD x']
reactorVectory = reactorPos[1] - NCaptPD['IBD y']
reactorVectorz = reactorPos[2] - NCaptPD['IBD z']

# Getting projected vector for positron annihilation along IBD to reactor core
projectedVector = (reactorVectorx * NCaptPD['X difference'] + reactorVectory * NCaptPD['Y difference'] + reactorVectorz * NCaptPD['Z difference']) / np.sqrt(reactorVectorx * reactorVectorx + reactorVectory * reactorVectory + reactorVectorz * reactorVectorz)

# Printing preview of dataframe to terminal
print(NCaptPD.head(20))

# Gathering the average difference in each direction and the average total difference
x_dif = NCaptPD['X difference'].mean()
y_dif = NCaptPD['Y difference'].mean()
z_dif = NCaptPD['Z difference'].mean()
distance_moved = NCaptPD['Total difference'].mean()
total_diff = np.sqrt(x_dif*x_dif + y_dif*y_dif + z_dif*z_dif)
avgProjVector = projectedVector.mean()

# Printing to terminal
print("Average X difference: ", x_dif)
print("Average Y difference: ", y_dif)
print("Average Z difference: ", z_dif)
print("Average total vector: ", total_diff)
print("Average total vector projected along the vector from the IBD location towards the reactor core: ", avgProjVector)
print("Average total distance moved from IBD point:", distance_moved)

# Plots for verification
plt.figure()
plt.hist(NCaptPD['N x'])
plt.xlabel("X axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Neutron Captures vs X")

plt.figure()
plt.hist(NCaptPD['N y'])
plt.xlabel("Y axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Neutron Captures vs Y")

plt.figure()
plt.hist(NCaptPD['N z'])
plt.xlabel("Z axis (mm)")
plt.ylabel("Counts")
plt.title("Number of Neutron Captures vs Z")

plt.figure()
plt.hist(NCaptPD['X difference'], bins=100)
plt.xlabel("X axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Neutron locations in X")

plt.figure()
plt.hist(NCaptPD['Y difference'], bins=100)
plt.xlabel("Y axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Neutron locations in Y")

plt.figure()
plt.hist(NCaptPD['Z difference'], bins=100)
plt.xlabel("Z axis (mm)")
plt.ylabel("Counts")
plt.title("Difference between the IBD and Neutron locations in Z")

plt.show()