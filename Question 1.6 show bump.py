import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

@dataclass
class p:
    v: float = 0.5          #velocity
    Tplate: float = 100   #plate temperature   
    Tair: float = 20.0      #air temperature
    Tn: float = 220.0       #nozzle temperature

    k: float = 400         #thermal conductivity
    c: float = 1500        #specific heat capacity
    rho: float = 1040      #density

    hair: float = 10     #convection heat transfer coefficient air
    hplate: float = 5     #conduction heat transfer coefficient plate
    A: float = 0.001        #area
    Lmax: float = 0.2*np.pi #maximum length of the filament

#calculate the fractions with know paramters
alpha = p.k / (p.c * p.rho)  # thermal diffusivity
betaair = (3*p.hair)/(p.c*p.rho*p.A)
betaplate = p.hplate/(p.c*p.rho*p.A)

#Discertisation for the model of lines:
N = 100
dd = p.Lmax/(N-1)
d_nodes = np.linspace(0, p.Lmax, N)  

#Adjust the PDE to make it solvable by solve_ipv
def pde(t,u):
    dudt = np.zeros(N)  #placeholders for values of u at every location ds
    Lt = p.v * t

    #first we assign u[0] and dudt[0] based on the boundary conditions
    if Lt < dd:
        #at d=0, we still need to heat up the filament
        dudt[0] = 1e6 *(p.Tn - u[0])
    elif Lt < p.Lmax:
        #at d=0, the filament interacts with air
        d2u_dd2 = (u[1]-(2*dd*p.hair)/p.k *(u[0]-p.Tair) - 2*u[0] +u[1])/(dd**2)  
        dudt[0] = alpha*d2u_dd2 - betaair*(u[0]-p.Tair) - betaplate*(u[0]-p.Tplate)
    else:
        #at d=0, the filament is connected to the filament at d=N
        d2u_dd2 = (u[1]-2*u[0] + u[N-1])/(dd**2)
        dudt[0] = alpha*d2u_dd2 - betaair*(u[0]-p.Tair) - betaplate*(u[0]-p.Tplate)
    

    #For the other nodes the Lt does not influence the process
    for i in range(1, N-1):
        if d_nodes[i] > Lt:
            #Node has not been touched yet
            dudt[i] = 0
        elif d_nodes[i] < Lt and d_nodes[i+1] > Lt:
                #Node just touched by nozzle
                dudt[i] = 1e6 *(p.Tn - u[i])
        elif d_nodes[i+1] < Lt:
            dudt[i] = alpha* (u[i-1] - 2*u[i] + u[i+1])/(dd**2) - betaair*(u[i]-p.Tair) - betaplate*(u[i]-p.Tplate)

    #Now the last node
    if d_nodes[N-1] > Lt:
        #the filament at d=N has not yet been printed
        dudt[N-1] = 0
    elif d_nodes[N-1] <= Lt and d_nodes[N-1] + dd >= Lt:
        dudt[N-1] = 1e6 *(p.Tn - u[N-1])
    elif d_nodes[N-1]+dd < Lt:
        #if the filament at d=N has been printed, it is connected tot the filament at d=0
        d2u_dd2 = (u[0]-2*u[N-1] + u[N-2])/(dd**2)
        dudt[N-1] = alpha*d2u_dd2 - betaair*(u[N-1]-p.Tair) - betaplate*(u[N-1]-p.Tplate)
        
    return dudt

#define the initial condition, all nodes are at 0 degrees
u0 = np.full(N, p.Tplate) 

#define the simulation time, and timesteps. Choose a time longer than the printing time 
#and make sure there are more timesteps than d nodes
t_print = p.Lmax/p.v
t = np.linspace(0, 2*t_print, 500)

#solve, choose LSODA method as it does not use too many iterations,
#but will switch if the the problem is stiff, which is often the case when transforming the PDE
#using the method of lines into ODEs
sol = solve_ivp(pde, [0, t[-1]], u0, method='LSODA', max_step = dd/p.v, t_eval = t)
print(sol)


plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label = 'Start of path (s=0)', color='blue')
plt.plot(sol.t, sol.y[(N-1)//2], label = 'Middle of path (s=Lmax/2)', color='red')
plt.plot(sol.t, sol.y[N-1], label = 'End of path (s=Lmax)', color='green')
plt.axvline(x=t_print, color = 'r', linestyle = '--', label = "End of Printing")
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature at the start of the filament over time')
plt.title('Filament Temperature Distribution Over Time')
plt.legend()
plt.grid(True)
plt.savefig('filament_cooling_bump.png')
        
         
        




