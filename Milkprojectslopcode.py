
# -*- coding: utf-8 -*-
"""

Created on Sat Mar 29 10:55:23 2025

 

@author: Hargi

"""

import numpy as np

from numba import njit

#import cProfile

#import pstats

from collections import Counter
 

 

c = 3e8

h=6.62e-34

Areasides = .0003 # in m sq

Areatop = .0001 # in m sq

wavelngth = 650e-9

 

radiusofparticle= 1e-6

 

x = 2 * np.pi * radiusofparticle/wavelngth

# x is the distance size of the particle over the

#wavelength of light.

 

refrel = 1.46/1.33  #particles refractive index to milk and waters,

 

 

 

nang = 180 #degrees of possible incidence

 

 

lisrt=[] #empty list to collect values

 

max_steps = 1000

 

#--------------DOMAIN------------

ymin = xmin = -5

ymax = xmax = 5

zmin = 0  

zmax = 45

#-------------------------------
faces = [(0, xmin, "left"),
(0, xmax, "right"),
(1, ymin, "behind"),
(1, ymax, "infront"),
(2, zmin, "below"),
(2, zmax, "above"),]
 

 
# --------CONSTANTS

 

 

N = 3.4e9 # number of milk particles per drop.

   

 

mu_a= 0.001  # absorption coefficient [1/mm] from paper cited

# absorption index will be imported for visible range.

 

numberphotons=int(1e6) # number of photons I am throwing through the milk

 

#totaldistance = 0.0

 

#absor = []

 

 


   

   

 

 

#-------------------------------Function 1 Mie theory Calculation-------------\

 
def bhmie(x, refrel, nang):

 

       

    s1_1 = np.zeros(nang, dtype=np.complex128)
    s1_2 = np.zeros(nang, dtype=np.complex128)
    s2_1 = np.zeros(nang, dtype=np.complex128)
    s2_2 = np.zeros(nang, dtype=np.complex128)

   

 

    #if nang > 1000:

     #   print("error: nang > mxnang=1000 in bhmie")

      #  return

 

    if nang < 2:

         nang = 2

 

    pii = 4.0 * np.arctan(1.0)

    dx = x

    drefrl = refrel

    y = x * drefrl

    ymod = abs(y)

 

    # Terminate series after NSTOP terms (with logarithmic derivatives

    #from nmx downward)

    xstop = x + 4.0 * x**0.3333 + 2.0

    nmx = max(xstop, ymod) + 15.0

    nmx = np.fix(nmx)

    nstop = int(xstop)

 

    #if nmx > nmxx:

     #  print("error: nmx > nmxx=%f for |m|x=%f" % (nmxx, ymod))

      #  return

 

    dang = 0.5 * pii / (nang - 1)

    amu = np.arange(0.0, nang, 1)

    amu = np.cos(amu * dang)

 

    pi0 = np.zeros(nang)

    pi1 = np.ones(nang)
    nn = int(nmx) - 1

    d = np.zeros(nn + 1)

    for n in range(0, nn):

         en = nmx - n

        # For n = nn, d[nn] remains 0 initially, so the recurrence starts

        #from the top.

        # This recurrence assumes that d[nn] is set to 0.
         d[nn - n - 1] = (en / y) - (1.0 / (d[nn - n] + en / y))

   

    # Upward recurrence for Riccati-Bessel functions

    psi0 = chi1= np.cos(dx)

    psi1 = np.sin(dx)

    chi0 = -np.sin(dx)
    xi1 = psi1 - chi1 * 1j

    qsca = 0.0

    gsca = 0.0

    p_val = -1
    for n in range(0, nstop):

        en = n + 1.0

        fn = (2.0 * en + 1.0) / (en * (en + 1.0))

       

        # Calculate psi_n and chi_n

        psi = (2.0 * en - 1.0) * psi1 / dx - psi0

        chi = (2.0 * en - 1.0) * chi1 / dx - chi0

        xi = psi - chi * 1j

       

       

       

        # Compute Mie coefficients an and bn

        an = ((d[n] / drefrl) + en / dx) * psi - psi1

        an = an / (((d[n] / drefrl) + en / dx) * xi - xi1)

        bn = ((drefrl * d[n]) + en / dx) * psi - psi1

        bn = bn / (((drefrl * d[n]) + en / dx) * xi - xi1)
        if n > 0:
            an1 = an

            bn1 = bn

       

        qsca += (2.0 * en + 1.0) * (abs(an)**2 + abs(bn)**2)

        gsca += ((2.0 * en + 1.0) / (en * (en + 1.0))) * (np.real(an) *

                                                          np.real(bn) +

                                                          np.imag(an) *

                                                          np.imag(bn))

       

        if n > 0:

           gsca += (((en - 1.0) * (en + 1.0)) / en) * (np.real(an1) *

                                                        np.real(an) +

                                                        np.imag(an1) *

                                                        np.imag(an) +

                                                         np.real(bn1) *

                                                         np.real(bn) +

                                                         np.imag(bn1) *

                                                         np.imag(bn))

       

        # Calculate scattering intensity pattern for angles 0 to 90 deg.

    tau = en * amu * pi1 - (en + 1.0) * pi0

    s1_1 += fn * (an * pi1 + bn * tau)

    s2_1 += fn * (an * tau + bn * pi1)

       

        # For angles > 90 deg, using symmetry do the same in reverse
    p_val = -p_val

    s1_2 += fn * p_val * (an * pi1 - bn * tau)

    s2_2 += fn * p_val * (bn * pi1 - an * tau)

       

        # Update psi and chi for next recurrence

    psi0 = psi1

    psi1 = psi

    chi0 = chi1

    chi1 = chi

    xi1 = psi1 - chi1 * 1j

       

        # Update pi_n values for next n

    pi_temp = pi1.copy()

    pi1 = ((2.0 * en + 1.0) * amu * pi1 - (en + 1.0) * pi0) / en

    pi0 = pi_temp.copy()

           

    # Concatenate the two halves of the scattering functions

    s1 = np.concatenate((s1_1, s1_2[-2::-1]))

    s2 = np.concatenate((s2_1, s2_2[-2::-1]))

   

    gsca = 2.0 * gsca / qsca

    qsca = (2.0 / (dx * dx)) * qsca

    qext = (4.0 / (dx * dx)) * np.real(s1[0])

   

    # Calculate backscattering efficiency (ensuring correct dimensionality)

    qback = 4 * (abs(s1[2 * nang - 2]) / dx)**2
    ctheta = np.linspace(0, 1, s1.size)
    return s1, s2, qext, qsca, qback, gsca, ctheta

#qext = extinction

#qsca = q of scattering

#gsca = the scattering g

# ctheta = The cos theta of our direction vector.

 

results = bhmie(x, refrel, nang)

#if results is not None:

s1, s2, qext, qsca, qback, gsca, ctheta = results
#print(results)

    #print("Qext =", qext)

    #print("Qsca =", qsca)

    #print("Qback =", qback)

    #print("g is =", gsca)

 

 

#----------------------FUNCTION 2 light propagation -------------------------

@njit

def sample_step_length(mu_t):

   

   return -np.log(np.random.rand()) / mu_t

 

def sample_scattering_angles(g):

   
    rnd = np.random.rand()

    #if g == 0:

     #   cos_theta = 2 * rnd - 1

    #else:

       

    term = (1 - g**2) / (1 - g + 2*g*rnd)
    cos_theta = (1 + g**2 - term**2) / (2 * g)

    cos_theta = np.clip(cos_theta, -1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand()

    return cos_theta, sin_theta, phi

@njit
def update_direction(direction, cos_theta, sin_theta, phi):

   

   dx, dy, dz = direction

 

   

   if np.abs(dz) > 0.9999:

       ux = sin_theta * np.cos(phi)

       uy = sin_theta * np.sin(phi)

       uz = np.sign(dz) * cos_theta

   else:
       denom = np.sqrt(1 - dz**2)

       

       ux = sin_theta * (dx * dz * np.cos(phi) - dy * np.sin(phi)) / (

                denom) + dx * cos_theta

       

       uy = sin_theta * (dy * dz * np.cos(phi) + dx * np.sin(phi)) / (

                denom) + dy * cos_theta

       

       uz = -sin_theta * np.cos(phi) * denom + dz * cos_theta
       
       
   new_direction = np.array([ux, uy, uz])
   return new_direction / np.linalg.norm(new_direction)




exit_counters = Counter({
"left": 0,
"right": 0,
"behind": 0,
"infront":0,
"below": 0,
"above": 0,
})

def monte_carlo_photon(mu_a, mu_s, g, light, max_steps):
    mu_t = mu_a + mu_s # my extinction coefficient
    
    
    pos = np.array([2*np.random.randn(), 0.0, 35+2*np.random.randn()])
    
    # my random position with respect to my inital height and size of cuvette
    #along with magnitudes of light scattering and natural gaussian decay
    
    direction = np.array([0.0, 1.0, 0.0]) # starting pointing in the y direct
    
    trajectory = [pos.copy()] # copy my last position
    
    
    for step in range(max_steps):
        
        prevpos = pos.copy() # copy my position
        
        
        candidates = [] # holder for later
        
        
        s = sample_step_length(mu_t) # find my length I will move
        
        
        pos += s * direction # multiplt that by my direction unit vector
        
        
        trajectory.append(pos.copy()) # copy the trajectory 
        
        for axis, bound, name in faces:# For when I am on the last iteration
        
            d = direction[axis] # let d by my direction vector on a specific 
            # axis
            
            
            if d == 0: # if my d is zero I am parralell to that axis so it
            # can be ignored
                continue
            
            
            t = (bound - prevpos[axis]) / d # solve for component of 
            #the distance to the bounded wall
            
            
            if 0 < t <= s: # if the distance to the bounded wall is less
            # than zero it's in the wrong direction, if it's greater than
            
            # s it's too far of a step to leave the face
                candidates.append((t, name)) # append the candidates that pass
        if candidates:
            tmin, face = min(candidates, key=lambda x: x[0])
            
            # Take the minimum of which ever t value was in my list
            exit_counters[face] += 1 # which ever face had the smallest t value
            # count that as the face I exited. 
            
            
            return np.array(trajectory), False # return my next direction
        
        if np.random.rand() < (mu_a / mu_t): # if my photon absorbed
        
            return np.array(trajectory), True# than it's true it absorbed
        cost, sint, phi = sample_scattering_angles(g) # if nothing left
        # the material then recalculate my angles again
        direction = update_direction(direction, cost, sint, phi)
        # and from there update my direction from those new angles
        
        
        
    return np.array(trajectory), False

       



 

#--------------------------Returned Variables------------------------------

 

mu_s= qsca*N*np.pi*radiusofparticle**2  # scattering coefficient [1/mm]


g= gsca      # anisotropy factor (determines direction of scattering)

      

#------------------------------PLOT-------------------------------------------

 

#fig = plt.figure(figsize=(8,6))

 

#ax = fig.add_subplot(111, projection='3d')

 

 

for i in range(numberphotons):

   

   trajectory, absorbed = monte_carlo_photon(mu_a,mu_s,g, wavelngth,

                                              max_steps)
"""

    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])

   

ax.set_title("Photon Random Walk Trajectory")

 

ax.set_xlabel("X")

 

ax.set_ylabel("Y")

 

ax.set_zlabel("Z")

 

plt.show()

"""

print("Qext =", qext)

print("Qsca =", qsca)

print("Qback =", qback)

print("g is =", gsca)

print(exit_counters)

 


 

 

#Path length calculation for light exiting the front

#print(max(lisrt))

#print(len(lisrt))

 

#print("total average length of photon path that exited upfront is",

 #     max(lisrt)/infront)

   

   

print("scattering coefficient is",mu_s * 10) # times 10 to go from cm to mm

 

 

 

 

#-------------------------FUNCTION 3 Length & Intensity Calc------------------        

 

 

   # print("The intenstiy up front is ", infront * h * c / (Areasides * wavelngth),

    #  "Watts per meter")

 

    #print("The intenstiy from right is ", right * h * c / (Areasides * wavelngth),

     # "Watts per meter")

 

    #print("The intenstiy behind is ", behind * h * c / (Areasides * wavelngth),

     # "Watts per meter")

 

    #print("The intenstiy above is ", above * h * c / (Areasides * wavelngth),

     #3 "Watts per meter")

 

    #print("The intenstiy from Left is ", left * h * c / (Areasides * wavelngth),

     # "Watts per meter")

 

#    print("The intenstiy below is ", below * h * c / (Areasides * wavelngth),

 #     "Watts per meter")

     

   

   

   

   

   

   

   

   

   

 
""""
if __name__ == "__main__":

    profiler = cProfile.Profile()

    profiler.enable()

   

    main()  # Run my simulation

   

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('tottime')

    stats.print_stats(20)  # I print the top 20 functions by total time

 """
 

 