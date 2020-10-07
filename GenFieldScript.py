# script to generate magnetization fields from enery and geometry parameters
# uses code from UbermagSANS.py to generate fields.
# saves the fields to a folder on remote server
import UbermagSANS as U
import numpy as np
import math
import scipy
from datetime import date

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper
# Def parameters for energy and geometry
Ms = 3.84e5
A = 8.78e-12
D = 1e-3
K = 0
u = (0, 0, 1)
xtalClass = 'T'

L = 2.5e-6
cell = (50e-9, 50e-9, 1e-9)
p1 = (-L/2, -L/2, 0)

h_max = 1.0
h_min = 0

t_max = 8
t_min = 1

iters = 50

for i in range(iters):
    a = np.random.choice(np.linspace(h_min, h_max,iters))
    b = np.random.choice(np.linspace(.1,1,10))

    h = a* 118010
    H = (0, 0, h)
    Ep = (Ms, A, D, K, u, H)

    t = b*1.2e-7
    p2 = (L/2, L/2, t)
    Gp = (L, t, cell, p1,p2)

    field = U.createField(Ep, Gp, xtalClass=xtalClass, show=False)


    d = date.today().strftime("%d_%m_%Y")
    name = 'a'+str(truncate(a,2))+'b'+str(truncate(b,2))
    U.saveField(field, name, 'fields'+d)
