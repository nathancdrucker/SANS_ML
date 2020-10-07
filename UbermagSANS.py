import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
import math
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as mc
plt.style.use('Solarize_Light2')

def createField(Ep, Gp, xtalClass, drive=True, show=True):
    # Input parameters for Hamiltonian Ep, geometric parameters Gp
    # output system.m object to be manipulated with other functions.
    Ms, A, D, K, u, H = Ep
    L, t, cell, p1, p2 = Gp
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell, bc='xy')
    system = mm.System(name='Mfield')
    system.energy = mm.Exchange(A=A)+mm.DMI(D=D,crystalclass=xtalClass)+mm.Zeeman(H=H)+mm.UniaxialAnisotropy(K=K,u=u)

    def m_initial(point):
        import random
        a = random.choice([-1,1])
        return (0, 0, a)

    system.m = df.Field(mesh, dim=3, value=m_initial, norm=Ms)
    if drive==True:
        md = mc.MinDriver()
        md.drive(system)
    if show==True:
        system.m.plane('z').mpl(figsize=(9, 7))
    return(system.m)


def theta(x):
    # x is 2D array
    # outputs matrix of angles in same shape as array
    x0, y0 = int(len(x)/2), int(len(x)/2)
    theta = np.zeros_like(x)
    for i in range(len(theta)):
        for j in range(len(theta)):
            theta[i][j] = np.arctan((i-y0)/ (j-x0+1e-6))
    return(theta)

def plotComps(field, L):
    n = np.shape(field.array)[1]
    mx = field.x.array[:,:,0,:].reshape(n,n)
    my = field.y.array[:,:,0,:].reshape(n,n)
    mz = field.z.array[:,:,0,:].reshape(n,n)

    mxft = abs(np.fft.fft2(mx))
    mxft = np.fft.fftshift(mxft)
    myft = abs(np.fft.fft2(my))
    myft = np.fft.fftshift(myft)
    mzft = abs(np.fft.fft2(mz))
    mzft = np.fft.fftshift(mzft)


    fig, ax = mpl.pyplot.subplots(2,3, figsize = [18,8])
    MX = ax[0,0].imshow(mx, origin='lower',cmap='RdBu',extent= [-L/2, L/2, -L/2, L/2],
        norm = colors.DivergingNorm(vmin=mx.min(), vcenter=0, vmax=mx.max()))
    ax[0,0].set_xlabel('x (nm)')
    ax[0,0].set_ylabel('y (nm)')
    colorbar(MX, title=r'$M_x$')
    ax[0,0].set_title(r'$M_x$')

    MY=ax[0,1].imshow(my, origin='lower',cmap='RdBu',extent= [-L/2, L/2, -L/2, L/2],
        norm = colors.DivergingNorm(vmin=mx.min(), vcenter=0, vmax=mx.max()))
    ax[0,1].set_xlabel('x (nm)')
    ax[0,1].set_ylabel('y (nm)')
    ax[0,1].set_title(r'$M_y$')
    colorbar(MY, title=r'$M_y$')

    MZ=ax[0,2].imshow(mz, origin='lower',cmap='RdBu',extent= [-L/2, L/2, -L/2, L/2],
        norm = colors.DivergingNorm(vmin=mx.min(), vcenter=0, vmax=mx.max()))
    ax[0,2].set_xlabel('x (nm)')
    ax[0,2].set_ylabel('y (nm)')
    ax[0,2].set_title(r'$M_z$')
    colorbar(MZ, title=r'$M_z$')

    MXFT = ax[1,0].imshow(mxft, origin='lower',cmap='Reds',extent=[-L/2, L/2, -L/2, L/2])
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title(r'$\tilde{M}_x$')
    colorbar(MXFT, title=r'$\tilde{M}_x$')

    MYFT = ax[1,1].imshow(myft, origin='lower',cmap='Reds',extent=[-L/2, L/2, -L/2, L/2])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_title(r'$\tilde{M}_y$')
    colorbar(MYFT, title=r'$\tilde{M}_y$')

    MZFT = ax[1,2].imshow(mzft, origin='lower',cmap='Reds',extent=[-L/2, L/2, -L/2, L/2])
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    ax[1,2].set_title(r'$\tilde{M}_z$')
    colorbar(MZFT, title=r'$\tilde{M}_z$')



def crossSection(field):
    n = np.shape(field.array)[1]
    mx = field.x.array[:,:,0,:].reshape(n,n)
    my = field.y.array[:,:,0,:].reshape(n,n)
    mz = field.z.array[:,:,0,:].reshape(n,n)

    mxft = abs(np.fft.fft2(mx))
    mxft = np.fft.fftshift(mxft)
    myft = abs(np.fft.fft2(my))
    myft = np.fft.fftshift(myft)
    mzft = abs(np.fft.fft2(mz))
    mzft = np.fft.fftshift(mzft)
    T = np.fft.fftshift(np.real(np.fft.fft2(mz)*np.conj(np.fft.fft2(my))+np.fft.fft2(my)*np.conj(np.fft.fft2(mz))))
    cross = mxft**2+myft**2*np.cos(theta(myft))**2+mzft**2*np.sin(theta(mzft))**2-T*np.sin(theta(T))*np.cos(theta(T))
    return(cross)

def plotCross(field, zoom=True):
    n = np.shape(field.array)[1]
    mx = field.x.array[:,:,0,:].reshape(n,n)
    my = field.y.array[:,:,0,:].reshape(n,n)
    mz = field.z.array[:,:,0,:].reshape(n,n)

    mxft = abs(np.fft.fft2(mx))
    mxft = np.fft.fftshift(mxft)
    myft = abs(np.fft.fft2(my))
    myft = np.fft.fftshift(myft)
    mzft = abs(np.fft.fft2(mz))
    mzft = np.fft.fftshift(mzft)
    fig, ax = mpl.pyplot.subplots(1,5, figsize= [28,4])
    MXFT = ax[0].imshow(mxft, origin='lower',cmap='Reds')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(r'$\tilde{M}_x$')
    colorbar(MXFT, title='Intensity')

    MYFT = ax[1].imshow(myft, origin='lower',cmap='Reds')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title(r'$\tilde{M}_y$')
    colorbar(MYFT, title='Intensity')

    MZFT = ax[2].imshow(mzft, origin='lower',cmap='Reds')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title(r'$\tilde{M}_z$')
    colorbar(MZFT, title='Intensity')

    T = np.fft.fftshift(np.real(np.fft.fft2(mz)*np.conj(np.fft.fft2(my))+np.fft.fft2(my)*np.conj(np.fft.fft2(mz))))
    TP = ax[3].imshow(abs(T), origin='lower',cmap='Reds')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title(r'$\tilde{T}_{yz}$')
    colorbar(TP, title='Intensity')

    cross = mxft**2+myft**2*np.cos(theta(myft))**2+mzft**2*np.sin(theta(mzft))**2-T*np.sin(theta(T))*np.cos(theta(T))
    crossP = ax[4].imshow(cross, origin='lower',cmap='jet')
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_title(r'$\frac{d\sigma}{d\Omega}$')
    colorbar(crossP, title='Intensity')

    if zoom==True:
        for axis in ax:
            axis.set_xlim(n/2-n/4, n/2+n/4)
            axis.set_ylim(n/2-n/4, n/2+n/4)

def lineCut(a, phi, n=250, show=True):
    import scipy.ndimage
    phi = math.pi*phi/180
    # a is square 2D array, phi is scalar
    r = int(len(a)/6)
    x0,y0 = int(len(a)/2), int(len(a)/2)
    x1,y1 = x0+r*np.cos(phi), x0+r*np.sin(phi)
    x, y = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    zi = scipy.ndimage.map_coordinates(a, np.vstack((x,y)))

    if show==True:
        fig, axes = mpl.pyplot.subplots(1,2,figsize=[10,5])
        xs = axes[0].imshow(a, origin='lower',cmap='jet')
        axes[0].plot([x0, x1], [y0, y1])
        axes[0].set_xlim([40,120])
        axes[0].set_ylim([40,120])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(r'$\frac{d\sigma}{d\Omega}$')
        colorbar(xs, title='Intensity')

        axes[1].plot(zi)
        axes[1].set_xlabel('q')
        axes[1].set_ylabel('I')

    return(zi)

def linecutAvg(a, phi1, phi2, n=100, show=True):
    tot = []

    for phi in np.linspace(phi1,phi2,n):
        tot.append(lineCut(a, phi, show=False))
    tot = np.array(tot)
    mu = np.mean(tot, axis=0)
    r = int(len(a)/6)
    x0,y0 = int(len(a)/2), int(len(a)/2)
    x1_1,y1_1 = x0+r*np.cos(phi1*math.pi/180), x0+r*np.sin(phi1*math.pi/180)
    x1_2,y1_2 = x0+r*np.cos(phi2*math.pi/180), x0+r*np.sin(phi2*math.pi/180)
    if show ==True:
        fig, axes = mpl.pyplot.subplots(1,2,figsize=[10,5])
        xs=axes[1].imshow(a, origin='lower',cmap='jet')
        axes[1].plot([x0, x1_1], [y0, y1_1],'g--')
        axes[1].plot([x0, x1_2], [y0, y1_2],'g--')
        axes[1].set_xlim([.25*len(a),.75*len(a)])
        axes[1].set_ylim([.25*len(a),.75*len(a)])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title(r'$\frac{d\sigma}{d\Omega}$')

        colorbar(xs, title='Intensity')

        axes[0].plot(mu)
        axes[0].set_xlabel('q')
        axes[0].set_ylabel('I')
    return(mu)

def SANS_Spectrum(Ep, Gp, xtalClass, drive=True):
    # takes in energy and geometry parameters Ep,GP
    # returns field, cross section, and lineCut
    field = createField(Ep, Gp, xtalClass, drive=drive, show=False)
    cross = crossSection(field)
    lc = linecutAvg(cross, 0, 180, show=False)
    return(field, cross, lc)


def colorbar(mappable, title):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, extend='both',shrink=0.5)
    cbar.ax.set_ylabel(title)
    plt.sca(last_axes)
    return cbar

def saveField(field,name,dir):
    # takes in discretisedfield.Field object with parameters used and saves it
    import os
    filename = name+'.omf'
    path = os.getcwd()+'/'+dir
    try:
        field.write(os.path.join(path,filename))
    except FileNotFoundError:
        os.makedirs(path)
        field.write(os.path.join(path,filename))

def openField(file):
    field = df.Field.fromfile(file)
    return(field)
