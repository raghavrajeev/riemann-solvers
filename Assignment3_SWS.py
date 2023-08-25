from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Heat capacity ratio
gamma=1.4

# Input
L=1.0
n=400
CFL=0.8
t=0.15
dx=L/n

# Left and right initial states
rho_Li,P_Li,u_Li=1.0,1.0,0.0
rho_Ri,P_Ri,u_Ri=0.125,0.1,0.0

e_Li=u_Li**2/2.0+P_Li/(rho_Li*(gamma-1.0))
e_Ri=u_Ri**2/2.0+P_Ri/(rho_Ri*(gamma-1.0))

# Initial state U and flux
U_cell=np.array([[rho_Li,rho_Li*u_Li,rho_Li*e_Li]]*int(n/2)\
                +[[rho_Ri,rho_Ri*u_Ri,rho_Ri*e_Ri]]*int(n/2))
U_cell_new=np.copy(U_cell)

F_cell_p=np.zeros([n,3])
F_cell_n=np.zeros([n,3])

K=np.zeros([3,3])
K_inv=np.zeros([3,3])
lamb_p=np.zeros([3,3])
lamb_n=np.zeros([3,3])
lamb=np.zeros([3])
A_p=np.zeros([3,3])
A_n=np.zeros([3,3])

itr=0
t_sim=0
lamb_max=0

# Iterating for time t
while(t_sim<t):
    
    # Iterating for n cells
    for i in range(n):
        
        # u,rho,P,e,H,a at each cell
        u=U_cell[i][1]/U_cell[i][0]
        rho=U_cell[i][0]
        P=(U_cell[i][2]/U_cell[i][0]-u**2/2)*(gamma-1)*rho
        
        e=u**2/2.0 + P/(rho*(gamma-1))
        H=(rho*e+P)/rho
        a=sqrt(gamma*P/rho)
        
        # Eigenvalues
        lamb=np.array([u-a,u,u+a])
        lamb_p=np.diag([max(l,0) for l in lamb])
        lamb_n=np.diag([min(l,0) for l in lamb])
        lamb_max=max(lamb_max,max(abs(lamb)))
        
        # Eigenvectors
        K[0]=[1,u-a,H-u*a]
        K[1]=[1,u,u**2/2]
        K[2]=[1,u+a,H+u*a]

        K=np.transpose(K)
        K_inv=np.linalg.inv(K)

        # A+ and A-
        A_p=K@lamb_p@K_inv
        A_n=K@lamb_n@K_inv
        
        # Flux in positive and negative direction
        F_cell_p[i]=A_p@U_cell[i]
        F_cell_n[i]=A_n@U_cell[i]
    
    dt=CFL*dx/lamb_max
    dt=min(dt,t-t_sim)
    
    # Updating U using fluxes
    for i in range(1,n-1):
        F_m=F_cell_p[i-1]+F_cell_n[i]
        F_p=F_cell_p[i]+F_cell_n[i+1]
        U_cell_new[i]=U_cell[i]-(dt/dx)*(F_p-F_m)
    
    U_cell=np.copy(U_cell_new)
    
    itr+=1
    t_sim+=dt
    lamb_max=0
    
# Plotting density, pressure, velocity and energy
x_min=0
x_max=L
x_plot=np.linspace(x_min,x_max,n)
rho_plot=np.array([Ui[0] for Ui in U_cell])
u_plot=np.array([Ui[1]/Ui[0] for Ui in U_cell])
P_plot=np.array([(Ui[2]/Ui[0]-(Ui[1]/Ui[0])**2/2)*(gamma-1)*Ui[0] for Ui in U_cell])
energy_plot=np.array([(Ui[2]/Ui[0]-(Ui[1]/Ui[0])**2/2) for Ui in U_cell])

# Importing and extracting analytic data
std_data=pd.read_csv("StandardSod_0_15.txt",sep=' ')
x_std=np.array(std_data['x'])
rho_std=np.array(std_data['rho'])
u_std=np.array(std_data['u'])
P_std=np.array(std_data['p'])
energy_std=np.array(std_data['ie'])

ax=[0]*4
for i in range(4):
    fig,ax[i]=plt.subplots(dpi=300)

ax[0].plot(x_std,rho_std,'k',label='Analytic')
ax[0].scatter(x_plot,rho_plot,marker='.',label='Steger-Warming')
ax[0].set_xlabel('Position')
ax[0].set_ylabel('Density')
ax[0].set_title('Density profile (Steger-Warming)')
ax[0].legend()

ax[1].plot(x_std,P_std,'k',label='Analytic')
ax[1].scatter(x_plot,P_plot,marker='.',label='Steger-Warming')
ax[1].set_xlabel('Position')
ax[1].set_ylabel('Pressure')
ax[1].set_title('Pressure profile (Steger-Warming)')
ax[1].legend()

ax[2].plot(x_std,u_std,'k',label='Analytic')
ax[2].scatter(x_plot,u_plot,marker='.',label='Steger-Warming')
ax[2].set_xlabel('Position')
ax[2].set_ylabel('Velocity')
ax[2].set_title('Velocity profile (Steger-Warming)')
ax[2].legend()

ax[3].plot(x_std,energy_std,'k',label='Analytic')
ax[3].scatter(x_plot,energy_plot,marker='.',label='Steger-Warming')
ax[3].set_xlabel('Position')
ax[3].set_ylabel('Energy')
ax[3].set_title('Energy profile (Steger-Warming)')
ax[3].legend()

plt.show()