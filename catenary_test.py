from pycatenary import cable
import numpy as np


# define properties of cable
length = 5  # length of line
w = 1.036  # submerged weight
EA = 560e3  # axial stiffness
floor = True  # if True, contact is possible at the level of the anchor
anchor = [0., 0., 0.]
fairlead = [2.51, 0., 2.5]

# create cable instance
l1 = cable.MooringLine(L=length,
                       w=w,
                       EA=EA,
                       anchor=anchor,
                       fairlead=fairlead,
                       floor=floor)

# compute calculations
l1.computeSolution()

for s in np.linspace(0,length):
    print(l1.s2xyz(s))