from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Heat capacity ratio
gamma=1.4
eps=1e-6

# Input
L=1.0
n=400
CFL=0.8
t=0.15
dx=L/n

# Left and right initial states
rho_L,P_L,u_L=rho_Li,P_Li,u_Li=1.0,1.0,0.0
rho_R,P_R,u_R=rho_Ri,P_Ri,u_Ri=0.125,0.1,0.0

e_L=e_Li=u_Li**2/2.0+P_Li/(rho_Li*(gamma-1.0))
e_R=e_Ri=u_Ri**2/2.0+P_Ri/(rho_Ri*(gamma-1.0))

# Initial state U and flux
U_cell=np.array([[rho_Li,rho_Li*u_Li,rho_Li*e_Li]]*int(n/2)\
                +[[rho_Ri,rho_Ri*u_Ri,rho_Ri*e_Ri]]*int(n/2))
U_cell_new=np.copy(U_cell)

F_cell=np.array([[rho_Li*u_Li,rho_Li*u_Li**2+P_Li,u_Li*(rho_Li*e_Li+P_Li)]]*int(n/2)\
                +[[rho_Ri*u_Ri,rho_Ri*u_Ri**2+P_Ri,u_Ri*(rho_Ri*e_Ri+P_Ri)]]*int(n/2))
F_bound=np.zeros([n-1,3])

K=np.zeros([3,3])
lamb=np.zeros([3])
delta=np.zeros([3])

itr=0
t_sim=0
lamb_max=0
# Iterating for time t
while(t_sim<t):
    
    # Iterating for n-1 interfaces
    for i in range(n-1):
        
        # rho,u,P,e,H at both sides of interface
        rho_L=U_cell[i][0]
        u_L=U_cell[i][1]/U_cell[i][0]
        P_L=(U_cell[i][2]/U_cell[i][0]-u_L**2 /2)*(gamma-1)*rho_L
        
        rho_R=U_cell[i+1][0]
        u_R=U_cell[i+1][1]/U_cell[i+1][0]
        P_R=(U_cell[i+1][2]/U_cell[i+1][0]-u_R**2 /2)*(gamma-1)*rho_R
        
        e_L=u_L**2/2+P_L/(rho_L*(gamma-1))
        e_R=u_R**2/2+P_R/(rho_R*(gamma-1))
        
        H_L=(rho_L*e_L+P_L)/rho_L
        H_R=(rho_R*e_R+P_R)/rho_R
        
        # tilda values
        u_til=(sqrt(rho_L)*u_L + sqrt(rho_R)*u_R)/(sqrt(rho_L) + sqrt(rho_R))
        H_til=(sqrt(rho_L)*H_L + sqrt(rho_R)*H_R)/(sqrt(rho_L) + sqrt(rho_R))
        a_til=(gamma-1)*(H_til-u_til**2/2)
        
        # Eigenvalues and finding max eigenvalue out of all interfaces
        lamb=np.array([u_til-a_til,u_til,u_til+a_til])
        lamb_max=max(lamb_max,max(max(lamb),-min(lamb)))
        
        # Eigenvectors
        K[0]=[1,u_til-a_til,H_til-u_til*a_til]
        K[1]=[1,u_til,u_til**2/2]
        K[2]=[1,u_til+a_til,H_til+u_til*a_til]
        
        # delta=beta-alpha
        delta[1]=(gamma-1)/(a_til**2) * ((rho_R-rho_L)*(H_til-u_til**2)\
                                         +u_til*(rho_R*u_R-rho_L*u_L)-(rho_R*e_R-rho_L*e_L))
        delta[0]=(1/(2*a_til))*((rho_R-rho_L)*(u_til+a_til) - (rho_R*u_R-rho_L*u_L) \
                                - a_til*delta[1])
        delta[2]=(rho_R-rho_L)-(delta[0]+delta[1])
        
        # Flux at each cell boundary is calculated and stored
        F_bound[i]=np.copy(F_cell[i])
        
        # Entropy fix is implemented
        if(abs(lamb[0])<eps):
            lamb[0]=0.5*(lamb[0]**2/eps + eps)
        if(abs(lamb[2])<eps):
            lamb[2]=0.5*(lamb[2]**2/eps + eps)
        for j in range(3):
            if(lamb[j]>0):
                break
            F_bound[i]+=lamb[j]*delta[j]*K[j]
    
    # Updating U using fluxes
    dt=CFL*dx/lamb_max
    dt=min(dt,t-t_sim)
    for i in range(1,n-1):
        U_cell_new[i]=U_cell[i]+(dt/dx)*(F_bound[i-1]-F_bound[i])
    
    U_cell=np.copy(U_cell_new)
    
    # Updating cell flux for updated U
    for i in range(n):
        rho_upd=U_cell[i][0]
        u_upd=U_cell[i][1]/U_cell[i][0]
        P_upd=(U_cell[i][2]/U_cell[i][0]-u_upd**2/2)*(gamma-1)*rho_upd
        e_upd=u_upd**2/2+P_upd/(rho_upd*(gamma-1))
        F_cell[i]=np.array([rho_upd*u_upd,rho_upd*u_upd**2+P_upd,u_upd*(rho_upd*e_upd+P_upd)])
    
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
energy_plot=np.array([Ui[2]/Ui[0]-(Ui[1]/Ui[0])**2/2 for Ui in U_cell])

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
ax[0].scatter(x_plot,rho_plot,marker='.',label='Roe solver (entropy fix)')
ax[0].set_xlabel('Position')
ax[0].set_ylabel('Density')
ax[0].set_title('Density profile (Roe with entropy fix)')
ax[0].legend()

ax[1].plot(x_std,P_std,'k',label='Analytic')
ax[1].scatter(x_plot,P_plot,marker='.',label='Roe solver (entropy fix)')
ax[1].set_xlabel('Position')
ax[1].set_ylabel('Pressure')
ax[1].set_title('Pressure profile (Roe with entropy fix)')
ax[1].legend()

ax[2].plot(x_std,u_std,'k',label='Analytic')
ax[2].scatter(x_plot,u_plot,marker='.',label='Roe solver (entropy fix)')
ax[2].set_xlabel('Position')
ax[2].set_ylabel('Velocity')
ax[2].set_title('Velocity profile (Roe with entropy fix)')
ax[2].legend(loc='upper left')

ax[3].plot(x_std,energy_std,'k',label='Analytic')
ax[3].scatter(x_plot,energy_plot,marker='.',label='Roe solver (entropy fix)')
ax[3].set_xlabel('Position')
ax[3].set_ylabel('Energy')
ax[3].set_title('Energy profile (Roe with entropy fix)')
ax[3].legend()

plt.show()