# Time-stepper for the glucose control toy model. By L. van Veen, Ontario Tech U, 2026.
# MCSC6120/MATH4041 Special Topics: Qualitative Analysis of Dynamical Systems.
import numpy as np
import matplotlib.pyplot as plt

nDof = 2
nPar = 4

# Time derivatives of the model (non-dimensionalized). In: current state (u,v) (float of shape (nDof,)) and parameters p (float of shape (nPar,)).
# Out: derivatives r (float of shape (nDof,))
def rhs(u,p):
    r = np.zeros((nDof,))
    f = -u[1]-u[0]*(u[1]+1.0)+p[3]
    r[0] = -p[0]*u[0]+p[1]*f+p[2]*u[1]
    r[1] = f
    return r

# Jacobian. In: current state (u,v) (float of shape (nDof,)) and parameters p (float of shape (nPar,)).
# Out: J (float of shape (nDof,nDof)).
def Jac(u,p):
    J = np.zeros((nDof,nDof))
    dfdu0 = -u[1]-1.0
    dfdu1 = -1.0 - u[0]
    J[0,0] = -p[0]+p[1]*dfdu0
    J[0,1] = p[1]*dfdu1+p[2]
    J[1,0] = dfdu0
    J[1,1] = dfdu1
    return J

# Second-order accurate Runge-Kutta step. In: current state (u,v) (float of shape (nDof,)), time step h (float), and parameters p (float of shape (nPar,)).
# Out: new state y2 (float of shape (nDof,)).
def RK2step(u,h,p):
    y0 = u
    f = rhs(y0,p)
    y1 = y0 + 0.5 * h * f
    f = rhs(y1,p)
    y2 = y0 + h * f
    return y2

# Set parameters: time step, system parameters.
h = 0.001
p = np.zeros((nPar,))
p[0] = 0.1
p[1] = 1.0
p[2] = 1.0
p[3] = 1.6

# Compute the discriminant in the equation for equilibria and print the result.
D = (p[0]+p[2])**2 + 4.0*p[0]*p[2]*p[3]
print('Discriminant is D=%f.' % (D))
if D>0:
    # If the equilibria exist, print their location and eigenvalues:
    eq0 = [-(p[0]+p[2])/(2.0*p[0])+np.sqrt(D)/(2.0*p[0]),-(p[0]+p[2])/(2.0*p[2])+np.sqrt(D)/(2.0*p[2])]
    J = Jac(eq0,p)
    L0,V0 = np.linalg.eig(J)
    eq1 = [-(p[0]+p[2])/(2.0*p[0])-np.sqrt(D)/(2.0*p[0]),-(p[0]+p[2])/(2.0*p[2])-np.sqrt(D)/(2.0*p[2])]
    J = Jac(eq1,p)
    L1,V1 = np.linalg.eig(J)
    print('Eq 0 at (%f,%f) with eigenvalues' % (eq0[0],eq0[1]))
    print(f"{L0[0]:.3f}",f"{L0[1]:.3f}")
    print('Eq 1 at (%f,%f) with eigenvalues' % (eq1[0],eq1[1]))
    print(f"{L1[0]:.3f}",f"{L1[1]:.3f}")

# u0 = [-11,-1.29943] close to stable manifold for p=[0.1,1,1,2]
# Set initial condition:
u0 = [-8,-1.8]
t = 0.0
# Set number of steps:
nStep = 40000
# Initialize:
out = np.zeros((nStep+1,3))
out[0,:] = [t,u0[0],u0[1]]
u = u0
# Loop over time steps:
for step in range(nStep):
    u = RK2step(u,h,p)
    t += h
    out[step+1,:] = [t,u[0],u[1]]
    if u[1] < -100.0: # Here you can specify some value for v at which the time-stepping terminates.
        print('Glucose too small, exiting...')
        break
# Plot the orbit with, if available, the equilibria and their eigenvectors (green for stable, red for unstable):
plt.plot(out[:step,1],out[:step,2],'-k')
if D>0:
    plt.plot(eq0[0],eq0[1],'*')
    if L0[0] < 0:
        col = 'g'
    else:
        col = 'r'
    plt.plot([eq0[0]-V0[0,0],eq0[0]+V0[0,0]],[eq0[1]-V0[1,0],eq0[1]+V0[1,0]],col)
    if L0[1] < 0:
        col = 'g'
    else:
        col = 'r'
    plt.plot([eq0[0]-V0[0,1],eq0[0]+V0[0,1]],[eq0[1]-V0[1,1],eq0[1]+V0[1,1]],col)
    plt.plot(eq1[0],eq1[1],'*')
    if L1[0] < 0:
        col = 'g'
    else:
        col = 'r'
    plt.plot([eq1[0]-V1[0,0],eq1[0]+V1[0,0]],[eq1[1]-V1[1,0],eq1[1]+V1[1,0]],col)
    if L1[1] < 0:
        col = 'g'
    else:
        col = 'r'
    plt.plot([eq1[0]-V1[0,1],eq1[0]+V1[0,1]],[eq1[1]-V1[1,1],eq1[1]+V1[1,1]],col)    
plt.show()
