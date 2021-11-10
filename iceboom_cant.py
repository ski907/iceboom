import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pycatenary import cable

# Plot limits: ideally the figure will not have to rescale during the animation
xmin, xmax = -5, 20
ymin, ymax = -15, 10
width, height = xmax - xmin, ymax - ymin

# The area density of sea water and ice.
rho_water, rho_ice = 1.027, 0.5
# Acceleration due to gravity, m.s-2
g = 9.81

def canonicalize_poly(xy):
    """Shift the (N+1,2) array of coordinates xy to start at minimum y."""
    #if get_area(xy) > 0:
    #    xy = xy[::-1]
    idx_ymin = xy[:,1].argmin()
    xy = np.roll(xy[:-1], -idx_ymin, axis=0)
    return np.vstack((xy, xy[0]))

def canonicalize_poly_index(xy,xy_index):
    """Shift the (N+1,2) array of coordinates xy to start at minimum y."""
    #if get_area(xy) > 0:
    #    xy = xy[::-1]
    idx_ymin = xy[:,1].argmin()
    xy = np.roll(xy[:-1], -idx_ymin, axis=0)
    xy_index = np.roll(xy_index[:-1], -idx_ymin, axis=0)
    return np.vstack((xy, xy[0])), np.append(xy_index, xy_index[0])

def rotate_poly(xy, xy_index, theta):
    """Rotate the (N+1,2) array of coordinates xy by angle theta about (0,0).

    The rotation angle, theta, is in radians.

    """
    s, c = np.sin(theta), np.cos(theta)
    R = np.array(((c, -s), (s, c)))
    xyp = (R @ xy.T).T
    return canonicalize_poly_index(xyp,xy_index)

def get_area(xy):
    """Return the area of the polygon xy.

    xy is a (N+1,0) NumPy array defining the N polygon vertices, but repeating
    the first vertex as its last element. The "shoelace algorithm" is used.

    """

    x, y = xy.T
    return np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) / 2

def get_cofm(xy, A=None):
    """Return the centre of mass of the polygon xy.

    xy is a (N+1,0) NumPy array defining the N polygon vertices, but repeating
    the first vertex as its last element. If the polygon area is not passed
    in as A it is calculated. The polygon must have uniform density.

    """

    if A is None:
        A = get_area(xy)
    x, y = xy.T
    Cx = np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1])) / 6 / A
    Cy = np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1])) / 6 / A
    return np.array((Cx, Cy))

def get_moi(xy, rho):
    """Return the moment of inertia of the polygon xy with density rho.

    xy is a (N+1,0) NumPy array defining the N polygon vertices, but repeating
    the first vertex as its last element.

    """

    x, y = xy.T
    Ix = rho * np.abs(np.sum((x[:-1]*y[1:] - x[1:]*y[:-1]) *
                (y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)) / 12)
    Iy = rho * np.abs(np.sum((x[:-1]*y[1:] - x[1:]*y[:-1]) *
                (x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2)) / 12)
    # Perpendicular axis theorem.
    Iz = Ix + Iy
    return Ix, Iy, Iz

def get_zero_crossing(pts):
    """Return the coordinates of the zero-crossing in pts.

    pts is a pair of (x, y) points, assumed to be joined by a straight line
    segment. This function returns the coordinates (x,0) at which this line
    crosses the y-axis (corresponding to sea-level in our model).

    """
    P0, P1 = pts
    x0, y0 = P0
    x1, y1 = P1
    if (x1-x0) == 0:
        return x1, 0
    m = (y1-y0)/(x1-x0)
    c = y1 - m*x1
    return -c/m, 0

def get_displaced_water_poly(iceberg, submerged=None):
    """Get the polygon for the submerged portion of the iceberg.

    iceberg is a (N+1,2) array of coordinates corresponding to the iceberg's
    vertexes in its current position and orientation (the first vertex is
    repeated at the end of the array);
    submerged is a boolean array corresponding to the vertexes which are
    under water (<0); if not provided it is calculated.

    """

    if submerged is None:
        submerged = (iceberg[:,1] <= 0)
    nsubmerged = sum(submerged)

    # Partially-submerged iceberg: find where it enters the sea, i.e. which
    # edges cross zero. zc_idx holds the indexes of the vertices *before*
    # each zero-crossing edge.
    diff = np.diff(submerged)
    zc_idx = np.where(diff)[0]
    # Interpolate to find the coordinates of the zero crossing.
    ncrossings = len(zc_idx)
    # We're going to build a polygon for the shape of the displaced water,
    # i.e. the submerged part of the iceberg.
    displaced_water = np.empty((nsubmerged + ncrossings, 2))
    # Loop over the points *before* each crossing in pairs. NB if the
    # iceberg is partially submerged, len(zc_idx) is guaranteed to be even.
    assert not ncrossings % 2
    i = j  = 0
    for idx1, idx2 in zip(zc_idx[0::2], zc_idx[1::2]):
        # All the submerged vertices up to the upwards crossing.
        displaced_water[j:j+idx1-i+1] = iceberg[i:idx1+1]
        # Work out where the crossing vertex should be and add it.
        c = get_zero_crossing(iceberg[idx1:idx1+2])
        j += idx1 - i + 1
        displaced_water[j] = c
        j += 1

        # Now the downward crossing: all the unsubmerged vertices are
        # skipped, and an extra vertex at sea level is added.
        c = get_zero_crossing(iceberg[idx2:idx2+2])
        displaced_water[j] = c
        j += 1
        i = idx2 + 1
    # Copy across any remaining submerged vertexes to displaced_water.
    displaced_water[j:] = iceberg[i:]
    return displaced_water

def apply_friction(omega, dh):
    """Apply frictional forces to the angular and linear velocities."""

    # Hard friction: angular and linear velocities are immediately quenched.
    # after movement.
    # return 0, np.array((0,0))

    # Intermediate friction: reduce the velocities by some fraction.
    return omega * 0.5, dh * 0.6

def find_coordinate(xy, xy_index, index):
    return xy[xy_index==index][0]

def ice_interface(iceberg,ice_thickness,Fh_mag):
    ice_density = 0.92
    top_of_boom = max(iceberg[:,1])
    top_of_ice = ice_thickness*(1-ice_density)
    bottom_of_ice = -ice_thickness*ice_density
    Fh = np.array((Fh_mag,0))
    
    if top_of_boom>top_of_ice:
        H = np.array((0,((top_of_ice + bottom_of_ice)*0.5)))
    elif top_of_boom < bottom_of_ice:
        H = np.array((0,0))
        Fh = np.array((0,0))
    else:
        H = np.array((0,(top_of_boom + bottom_of_ice)*0.5))
        
    return H, Fh

def get_cable_coords(l1,cable_legnth):   
    cable_coords_list = []
    for s in np.linspace(0,cable_length):
        cable_coords_list.append([l1.s2xyz(s)[0],l1.s2xyz(s)[2]])
        cable_coords = np.array(cable_coords_list)
    return cable_coords

def check_if_catenary_valid(Z,anchor_point,cable_length):
    if abs(Z[1] - anchor_point[1]) + abs(Z[0] - anchor_point[0]) > cable_length*1.05:
        #print(abs(Z[1] - anchor_point[1]) + abs(Z[0] - anchor_point[0]))
        cat_valid = True
    else:
        cat_valid = False
    return cat_valid

def run_catenary(l1,Z,anchor_point,cable_length,cat_valid):
    if cat_valid: 
        l1.computeSolution()
        cable_coords = get_cable_coords(l1,cable_length)
    else:
        cable_coords = np.array((Z,
                                 (Z[0],anchor_point[1]),
                                 anchor_point))    
    return cable_coords, l1

    
# Our two-dimensional iceberg, defined as a polygon.
#poly = [
#(3,0), (3,3), (1,5), (4,7), (0,12), (1,15), (4,17), (6,14), (7,14), (8,12),
#(7,10), (7,7), (5,1)
#]

#poly = [
#(3,0), (7,0), (7,4), (3,4)
#]

poly=[]
radius = 1
for dtheta in np.linspace(0,2*np.pi,100):
    poly.append(np.array((radius*np.sin(dtheta),radius*np.cos(dtheta))))

Fz_mag = 0

# Repeat the initial vertex at the end, for convenience.
iceberg0 = np.array(poly + [poly[0]])
iceberg0_index = np.append(np.array(range(len(iceberg0)-1)),0)
# Centre the iceberg's local coordinates on its centre of mass.
iceberg0 = iceberg0 - get_cofm(iceberg0)
# We might want to start the iceberg off in some different orientation:
# if so, rotate it here.
iceberg0, iceberg0_index = rotate_poly(iceberg0, iceberg0_index, -0.2)

# Get the (signed) area, mass, and weight of the iceberg.
A = get_area(iceberg0)
M = rho_ice * abs(A)
Fg = np.array((0, -M * g))

# We also need the Iz component of the iceberg's moment of inertia.
_, _, Iz = get_moi(iceberg0, rho_ice)


#cable anchor
anchor_point = np.array((7.5,-2))

load_point = 25
Z = find_coordinate(iceberg0, iceberg0_index, load_point)

Fh_mag = -5
boom_under_ice = False
boom_ice_friction = 0.2
#Fi = np.array((0,0))

cable_length = 10
cable_w = .5  # submerged weight
cable_EA = 560e3  # axial stiffness
cable_floor = False  # if True, contact is possible at the level of the anchor
cable_anchor = [anchor_point[0], 0., anchor_point[1]]
cable_fairlead = [Z[0], 0., Z[1]]


total_cable_weight = cable_length * cable_w

# create cable instance
l1 = cable.MooringLine(L=cable_length,
                       w=cable_w,
                       EA=cable_EA,
                       anchor=cable_anchor,
                       fairlead=cable_fairlead,
                       floor=cable_floor)

# compute calculations
cat_valid = check_if_catenary_valid(Z,anchor_point,cable_length)
if cable_floor == False:
    cat_valid = True
cable_coords, l1 = run_catenary(l1,Z,anchor_point,cable_length,cat_valid)

fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# We would prefer equal distances in the x- and y-directions to look the same.
ax.axis('equal')

# The centre of mass starts at this height above sea level.
h = 0
# theta is the turning angle of the iceberg from its initial orientation;
# G is the position of its centre of mass (in world coordinates).
theta, G = 0, np.array((0, h))
# omega = dtheta / dt is the angular velocity; dh is the linear velocity.
omega, dh = 0, np.array((0,0))
# The time step (s): small, but not too small.
dt = 0.1

phi = 0
def update(it):
    """Update the animation for iteration number it."""

    global omega, dh, G, theta, Fz_mag, Fh_mag, l1, boom_under_ice

    print('iteration #{}'.format(it))

    # Update iceberg orientation and position.
    theta += omega * dt
    G = G + dh * dt
    


    # Rotate and translate a copy of the original iceberg into its current.
    # position.
    iceberg, iceberg_index = rotate_poly(iceberg0, iceberg0_index, theta)
    iceberg = iceberg + G
    
    ice_thickness = .5
    
    H, Fh = ice_interface(iceberg,ice_thickness,Fh_mag)

    #find location of load point Z

    Z = find_coordinate(iceberg, iceberg_index, load_point)
    
    #update the location of where the chain is attached
    l1.setFairleadCoords([Z[0], 0., Z[1]])
    cat_valid = check_if_catenary_valid(Z,anchor_point,cable_length)
    if cable_floor == False:
        cat_valid = True
    cable_coords, l1 = run_catenary(l1,Z,anchor_point,cable_length,cat_valid)
    
   
    cable_vector = anchor_point - Z
    
    unit_vector_1 = cable_vector / np.linalg.norm(cable_vector)
    unit_vector_2 = np.array((-80,0)) / np.linalg.norm(np.array((-100,0)))
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    anchor_theta = np.arccos(dot_product)
    
    #cable force
 
    
    comp_cable_length = np.linalg.norm(cable_vector)
    
    if comp_cable_length>=cable_length:
        cable_force = np.array((0.,-total_cable_weight))
        
       
        spring_k = 1000
        Fspring = spring_k * (comp_cable_length - cable_length) * unit_vector_1
        cable_force += Fspring
        
        #print(np.tan(anchor_theta) * (-Fh[0]))
    else:
        #cable_force = np.array((0,0))
        if cat_valid:
            cable_force = -l1.getTension(cable_length)[[0,2]]
        else:
            partial_cable_weight = l1.w*abs((Z[1]-anchor_point[1]))
            cable_force = np.array((0.,-partial_cable_weight[0]))

    Fz = cable_force
    print(Fz)
    
    # Which vertices are submerged (have their y-coordinate negative)?
    submerged = (iceberg[:,1] <= 0)
    nsubmerged = sum(submerged)

    iceberg_in_water = True
    if nsubmerged in (0, 1):
        # The iceberg is in the air above the surface of the sea.
        B = None
        Adisplaced = 0
        alpha = 0
        iceberg_in_water = False

    if iceberg_in_water:
        # Apply some frictional forces which damp the motion in water.
        omega, dh = apply_friction(omega, dh)
        if nsubmerged == len(submerged):
            # The iceberg is fully submerged.
            displaced_water = iceberg
            Adisplaced = A
            B = G
        else:
            displaced_water = get_displaced_water_poly(iceberg, submerged)

            # Area of the displaced water and position of the centre of buoyancy.
            Adisplaced = get_area(displaced_water)
            B = get_cofm(displaced_water)

    # Buoyant force due to the displaced water.
    Fb = np.array((0, rho_water * abs(Adisplaced) * g))
    
    Fi = np.array((0,0))
    I = iceberg[iceberg[:,1].argmax()]
    #ice friction
    if boom_under_ice:
        print("boom")
        Fi[0]= Fb[1] * boom_ice_friction * - np.sign(Fh_mag) 
        k_ice = 3000
        Fi[1] = k_ice*(-ice_thickness*0.92 - I[1])
        
        

        
        

    if B is not None:
        # Vector from G to B
        r = B - G
        # Torque about G
        tau = np.cross(r, Fb)
        alpha = tau / Iz
        print(Iz)
    
    #H = np.array((0,-ice_thickness/2))
    rh = H - G
    tau_h = np.cross(rh, Fh)
    alpha += tau_h / Iz
        
    rz = Z - G
    tau_z = np.cross(rz, Fz)
    alpha += tau_z / Iz
    
    
    ri = I - G
    tau_i = np.cross(ri, Fi)
    alpha += tau_i / Iz
    

    
    # Resultant force on the iceberg.
    F = Fg + Fb + Fz + Fh + Fi
    # Net linear acceleration.
    a = F / M

    # Now plot the scene for this frame of the animation.
    ax.clear()

    # The sea! The sea!
    sea_patch = plt.Rectangle((xmin, ymin), width, -ymin, fc='#8888ff')
    ax.add_patch(sea_patch)

    # The iceberg itself, in its current orientation and position.
    poly_patch = plt.Polygon(iceberg, fc='#ddddff', ec='k')
    ax.add_patch(poly_patch)

    if B is not None:
        # Draw the submerged part of the iceberg in a different colour.
        poly_patch = plt.Polygon(displaced_water, fc='#ffdddd', ec='k')
        ax.add_patch(poly_patch)

        # Indicate the position of the centre of buoyancy.
        bofm_patch = plt.Circle(B, 0.2, fc='b')
        ax.add_patch(bofm_patch)

    # Indicate the position of the centre of mass.
    cofm_patch = plt.Circle(G, 0.2, fc='r')
    ax.add_patch(cofm_patch)
    
    if iceberg_in_water:
        ice_density = 0.92
        top_of_ice = ice_thickness*(1-ice_density)
        bottom_of_ice = -ice_thickness*ice_density
        if nsubmerged == len(submerged):
            print("sub")
            if max(iceberg[:,1])<bottom_of_ice:
                boom_under_ice = True
                front_of_ice = iceberg[iceberg[:,1].argmax()][0]
            else:
                front_of_ice = iceberg[iceberg[:,1].argmax()][0]
        else:
            waterline = displaced_water[displaced_water[:,1]==0]
            if np.sign(Fh_mag)==1:
                front_of_ice = waterline[waterline[:,0].argmin()][0]
                
            else:
                front_of_ice = waterline[waterline[:,0].argmax()][0]
                #print(front_of_ice)

        back_of_ice = front_of_ice - np.sign(Fh_mag)*10
        
        ice_sheet = np.array([[front_of_ice,bottom_of_ice],[back_of_ice,bottom_of_ice],[back_of_ice,top_of_ice],[front_of_ice,top_of_ice]])
        ice_sheet_patch = plt.Polygon(ice_sheet, fc='#ffdddd',ec='k')
        ax.add_patch(ice_sheet_patch)
    
    if comp_cable_length >= cable_length:
        # Draw the cable
        the_cable = np.array((Z,anchor_point))
        cable_patch = plt.Polygon(the_cable, fc='#ffdddd', ec='k')
        ax.add_patch(cable_patch)
    else:
        #Draw catenary cable
        the_catenary_cable = np.array(cable_coords)
        #print(cable_coords)
        cat_cable_patch = plt.Polygon(the_catenary_cable, fc=None, fill=False, ec='k',closed=False)
        ax.add_patch(cat_cable_patch)
    
    # Update the angular and linear velocities
    omega += alpha * dt
    dh = dh + a * dt

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('equal')

ani = animation.FuncAnimation(fig, update, 150, blit=False, interval=100, repeat=True)
plt.show()


def main():
    i=0
    while True:
        update(i)
        i+=1
#    
#if __name__ == "__main__":
#    main()