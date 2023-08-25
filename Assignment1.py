import numpy as np
from numpy.linalg import eig,inv
import matplotlib.pyplot as plt

# User input 
m=int(input("Enter number of equations (m): "))
A=[]

print("Enter (space separated) row-wise input for matrix A:")
for i in range(m):
    a = [float(x) for x in input().split()]
    A.append(a)

U_L0=[float(x) for x in input("Enter U_L0 (m space separated values): ").split()]
U_R0=[float(x) for x in input("Enter U_R0 (m space separated values): " ).split()]
n=int(input("Enter number of cells (n): "))
L=float(input("Enter domain size (L) (in m): "))
t=float(input("Enter total simulation time (in seconds): "))
CFL=0.8

x_min=-L/2
x_max=L/2
dx=(x_max-x_min)/n

# Storing eigenvalues and eigenvectors
lamb=eig(A)[0]
K=eig(A)[1]

# Calculating time step
dt=CFL*dx/max(max(lamb),min(lamb))

# Transforming U to W
U0=([U_L0]*int(n/2))+([U_R0]*int(n/2))
U=U0
W=np.array([(inv(K)@i) for i in U])

lambda_p=[max(i,0) for i in lamb]
lambda_n=[min(i,0) for i in lamb]

t_total=0
iter=0
# Solve till waves travel for time t
while(t_total<t):
    # Loop for each wave
    for j in range(m):
        # Adjusting dt at last iteration so that t_total=t at end 
        dt=min(dt,t-t_total)
        # Solving upwind scheme equation
        for i in range(1,int(n/2)):
            W[i][j]=W[i][j] - (dt/dx)*lambda_n[j]*(W[i+1][j]-W[i][j]) \
                - (dt/dx)*lambda_p[j]*(W[i][j]-W[i-1][j])
        for i in reversed(range(int(n/2),n-1)):
            W[i][j]=W[i][j] - (dt/dx)*lambda_n[j]*(W[i+1][j]-W[i][j]) \
                - (dt/dx)*lambda_p[j]*(W[i][j]-W[i-1][j])
    iter+=1
    t_total+=dt

# Transforming back to U
U=np.array([K@i for i in W])

# W0=np.array([(inv(K)@i) for i in U0])

# Plotting U0
x=np.linspace(x_min,x_max,n)
U0=np.array(U0)
fig1,ax1=plt.subplots(dpi=300)
for i in range(m):
    ax1.plot(x,U0[:,i],label="U"+str(i+1))
ax1.legend()
ax1.set_title("U profile at initial condition")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("U")
ax1.grid(linewidth=0.5)

# Plotting U
fig,ax=plt.subplots(dpi=300)
for i in range(m):
    ax.plot(x,U[:,i],label="U"+str(i+1))
ax.legend()
ax.set_title("U profile after simulation time t")
ax.set_xlabel("x (m)")
ax.set_ylabel("U")
ax.grid(linewidth=0.5)