#***********************************************************************
#     2-D FDTD TE code with Berenger PML and Antenna
#***********************************************************************
#
#     Program author: Nicholas Muhlmeyer
#
#     Date of this version:  July 2016
#
#     An E-plane sectoral horn has been modeled in free space by
#     using copper material. The aperture faces the right PML boundary.
#
#     The source excitation is an magnetic sheet sheet current located
#     lambda/4 from the back wall of the waveguide.
#
#     To execute this function, type "myfdtd" at the MATLAB prompt. 
#     This function can displays the FDTD-computed Ex, Ey, and Hz fields
#     at every time step. 
#***********************************************************************
#
#
#*********************************************************************** 
#     2-D FDTD TE code with PML absorbing boundary conditions 
#*********************************************************************** 
# 
# 
#            ---------------------------------------------- 
#           |  |                BACK PML                |  | 
#            ---------------------------------------------- 
#           |L |                                       /| R| 
#           |E |                                (ib,jb) | I| 
#           |F |                                        | G| 
#           |T |                                        | H| 
#           |  |                MAIN GRID               | T| 
#           |P |                                        |  | 
#           |M |                                        | P| 
#           |L | (1,1)                                  | M| 
#           |  |/                                       | L| 
#            ---------------------------------------------- 
#           |  |                FRONT PML               |  | 
#            ---------------------------------------------- 
# 
#*********************************************************************** 

import math
import numpy
import scipy
import matplotlib
#*********************************************************************** 
#     Fundamental constants 
#*********************************************************************** 
def def_constants():
    global cc, muz,epsz,etaz,freq,freq,lambda1,omega,T
    cc=2.99792458e8             #speed of light in free space 
    muz=4.0*math.pi*1.0e-7      #permeability of free space 
    epsz=1.0/(cc*cc*muz)        #permittivity of free space 
    etaz = math.sqrt(muz/epsz)  #instrinsic impedance of free space
 
    freq = 10.0e9               #center frequency of source excitation 
    lambda1=cc/freq                #center wavelength of source excitation 
    omega=2.0*math.pi*freq
    T = 1/freq
    # freq = 12.0e9; %different excitation freq
    # omega = 2.0*pi*freq; %different excitation freq
    return

#*********************************************************************** 
#     Analysis Methods 
#*********************************************************************** 
def def_analysis():
    global batchfile, realtime, gridplot, softsource
    global efieldsource, ss, nff,twoantenna, backwall
    batchfile = 0
    realtime = 0            #updating time plots
    gridplot = 0            #plot materials in grid
    softsource = 1          #use soft source
    efieldsource = 1        #use impressed E field as source
    ss = 0                  #steady state sinusoid
    nff = 0                 #Find near to Far-field pattern
    twoantenna = 0          #use recieving antenna
    backwall = 0            #include backwall
    if batchfile == 1:
        ss = 0
        nff = 0
        twoantenna = 0
        backwall = 0
    return

#*********************************************************************** 
#     Grid parameters 
#*********************************************************************** 
def grid_parameters():
    global ie, je, ib, jb, dx, dt, nmax, iebc, jebc,rmax, orderbc
    global ibbc, jbbc, iefbc, ibfbc, n1, n2, fttbuffer
    if not batchfile:
        scale = 1
    ie = 400                #number of grid cells in x-direction 
    je = 400                #number of grid cells in y-direction 
    ib = ie + 1             #location of z-directed hard source 
    jb = je +1              #location of z-directed hard source
    dx = lambda1/20/scale   #space increment of square lattice 
    dt = dx/(2.0*cc)        #time step 
    if not batchfile:
        nmax = 200          #total number of time steps
    iebc = 8                #thickness of left and right PML region
    jebc = 8                #thickness of front and back PML region
    rmax = 1.0e-7
    orderbc = 2
    ibbc = iebc + 1
    jbbc = jebc + 1
    iefbc = ie + 2 * jebc
    ibfbc = iefbc + 1
    if ss:
        n1 = 4986
        n2 = n1 + T/(4*dt)  #steady state timestep T/4 later
    if nff:
        fttbuffer = 12      #buffer around boundary to run FFT
    return

#*********************************************************************** 
#     Material parameters 
#*********************************************************************** 
def def_material_parameters():
    global media, eps,sig,mur,sim
    media=2                     # media [free space, PEC]
    eps=[1.0,1.0]
    sig=[0.0,1e100]
    mur=[1.0,1.0]
    sim=[0.0,0.0]
    return

def_constants()
def_analysis()
grid_parameters()
def_material_parameters()

#*********************************************************************** 
#     Wave excitation 
#*********************************************************************** 

#source = fdtdsource(ss,freq,omega,dt,nmax,etaz)

#*********************************************************************** 
#     Field arrays 
#*********************************************************************** 
def field_arrays():
    global ex, ey, hz
    global exbcb, eybcb, hzxbcb, hzybcb
    global exbcf, eybcf,hzxbcf,hzybcf
    global exbcl, eybcl, hzxbcl, hzybcl
    global exbcr, eybcr, hzxgcr, hzybcr
    ex = numpy.zeros([ie,jb])               #fields in main grid
    ey = numpy.zeros([ie,jb])
    hz = numpy.zeros([ie,je])
    
    exbcf = numpy.zeros([iefbc,jebc])       #fields in front PML region
    eybcf = numpy.zeros([ibfbc,jebc])
    hzxbcf = numpy.zeros([iefbc,jebc])
    hzybcf = numpy.zeros([iefbc,jebc])
    
    exbcb = numpy.zeros([iefbc,jbbc])       #fields in back PML region
    eybcb = numpy.zeros([ibfbc,jebc])
    hzxbcb = numpy.zeros([iefbc,jebc])
    hzybcb = numpy.zeros([iefbc,jebc])
    
    exbcl = numpy.zeros([iebc,jb])          #fields in left PML region 
    eybcl = numpy.zeros([iebc,je])
    hzxbcl = numpy.zeros([iebc,je])
    hzybcl = numpy.zeros([iebc,je])
    
    exbcr = numpy.zeros([iebc,jb])          #fields in right PML region
    eybcr = numpy.zeros([ibbc,je])
    hzxgcr = numpy.zeros([iebc,je])
    hzybcr = numpy.zeros([iebc,je])
    return
 
#*********************************************************************** 
#     Array Initialization 
#*********************************************************************** 
def array_initialization():
    global ca, cb, da, db
    global caexbcf, cbexbcf, dahzbcf, dbhybcf
    global caexbcb, cbexbcb, dahzybcb, dbhzybcb
    global caeybcl, cbeybcl, dahzxbcl, dbhzxbcl
    global dahzxbcf, dbhzxbcf, dahzxbcb, dbhzxbcb
    global caexbcl, cbexbcl, dahzybcl, dbhzybcl
    global caeybcr, cbeybcr,caeybcf , cbeybcf
    global caeybcb, cbeybcb, dahzxbcr, dbhzxbcr
    global dahzybcr, dbhzybcr
    global N, eyr, hzr, sink1, sink2, sink3
    ca = [0.0 for x in range(media)]
    cb = [0.0 for x in range(media)]
    da = [0.0 for x in range(media)]
    db = [0.0 for x in range(media)]
    caexbcf = numpy.zeros([iefbc,jebc-1])
    cbexbcf = numpy.zeros([iefbc,jebc-1])
    dahzbcf = numpy.zeros([iefbc,jebc])
    dbhybcf = numpy.zeros([iefbc,jebc])
    caexbcb = numpy.zeros([iefbc,jebc-1])
    cbexbcb = numpy.zeros([iefbc,jebc-1])
    dahzybcb = numpy.zeros([iefbc,jebc])
    dbhzybcb = numpy.zeros([iefbc,jebc])
    caeybcl = numpy.zeros([iebc,je])
    cbeybcl = numpy.zeros([iebc,je])
    dahzxbcl = numpy.zeros([iebc,je])
    dbhzxbcl = numpy.zeros([iebc,je])
    dahzxbcf = numpy.zeros([iebc,jebc])
    dbhzxbcf = numpy.zeros([iebc,jebc])
    dahzxbcb = numpy.zeros([iebc,jebc])
    dbhzxbcb = numpy.zeros([iebc,jebc])
    caexbcl = numpy.zeros([iebc,je-1])
    cbexbcl = numpy.zeros([iebc,je-1])
    dahzybcl = numpy.zeros([iebc,je])
    dbhzybcl = numpy.zeros([iebc,je])
    caeybcr = numpy.zeros([iebc-1,je])
    cbeybcr = numpy.zeros([iebc-1,je])
    caeybcf = numpy.zeros([2*iebc-1+ie,jebc])
    cbeybcf = numpy.zeros([2*iebc-1+ie,jebc])
    caeybcb = numpy.zeros([2*iebc-1+ie,jebc])
    cbeybcb = numpy.zeros([2*iebc-1+ie,jebc])
    dahzxbcr = numpy.zeros([iebc,je])
    dbhzxbcr = numpy.zeros([iebc,je])
    dahzybcr = numpy.zeros([iebc,je])
    dbhzybcr = numpy.zeros([iebc,je])
    
    if nff:
        N = numpy.power(2,16)
        if ss:
            eyr = [0.0 for x in range(je/2 - fftbuffer + 1)]
            hzr = [0.0 for x in range(je/2 - fftbuffer + 1)]
    if batchfile:
        sink1 = numpy.zeros(nmax)
        sink2 = numpy.zeros(nmax)
    sink3 = numpy.zeros(nmax)
    
    return

#*********************************************************************** 
#     Updating coefficients 
#*********************************************************************** 
def updating_coefficients():
    global eaf, haf
    for i in range(media):
        eaf = dt*sig[i] / (2.0 * epsz * eps[1])
        ca[i] = (1.0 - eaf) / (1.0 + eaf)
        cb[i]= dt / (epsz * eps[i] * dx * (1.0 + eaf)) 
        haf = dt * sim[i] / (2.0 * muz * mur[i])
        da[i] = (1.0 - haf) / (1.0 + haf)
        db[i]= dt / (muz * mur[i] * dx * (1.0 + haf))
    return
print dt
field_arrays()
array_initialization()
updating_coefficients()
'''
%*********************************************************************** 
%     Geometry specification (main grid) 
%*********************************************************************** 

[caex cbex caey cbey dahz dbhz , ~, ~, b , ~, is js iread jread iread2 jread2] ...
    = gridspace(twoantenna,ca,cb,da,db,ib,jb,lambda,ie,je,dx,backwall);

%*********************************************************************** 
%     Plot Materials in Grid 
%*********************************************************************** 

if gridplot == true
    materialgrid(caex,caey,is,js,ca,ie,je);
end

%*********************************************************************** 
%     Fill the PML regions 
%*********************************************************************** 
 
delbc=iebc*dx; 
sigmam=-log(rmax)*(orderbc+1)/(2*etaz*delbc); 
bcfactor=sigmam/(dx*(delbc^orderbc)*(orderbc+1)); 
 
%     FRONT region  
 
caexbcf(1:iefbc,1)=1.0; 
cbexbcf(1:iefbc,1)=0.0; 
for j=2:jebc 
  y1=(jebc-j+1.5)*dx; 
  y2=(jebc-j+0.5)*dx; 
  sigmay=bcfactor*(y1^(orderbc+1)-y2^(orderbc+1)); 
  ca1=exp(-sigmay*dt/epsz); 
  cb1=(1.0-ca1)/(sigmay*dx); 
  caexbcf(1:iefbc,j)=ca1; 
  cbexbcf(1:iefbc,j)=cb1; 
end 
sigmay = bcfactor*(0.5*dx)^(orderbc+1); 
ca1=exp(-sigmay*dt/epsz); 
cb1=(1-ca1)/(sigmay*dx); 
caex(1:ie,1)=ca1; 
cbex(1:ie,1)=cb1; 
caexbcl(1:iebc,1)=ca1; 
cbexbcl(1:iebc,1)=cb1; 
caexbcr(1:iebc,1)=ca1; 
cbexbcr(1:iebc,1)=cb1; 
 
for j=1:jebc 
  y1=(jebc-j+1)*dx; 
  y2=(jebc-j)*dx; 
  sigmay=bcfactor*(y1^(orderbc+1)-y2^(orderbc+1)); 
  sigmays=sigmay*(muz/epsz); 
  da1=exp(-sigmays*dt/muz); 
  db1=(1-da1)/(sigmays*dx); 
  dahzybcf(1:iefbc,j)=da1; 
  dbhzybcf(1:iefbc,j)=db1; 
  caeybcf(1:ibfbc,j)=ca(1); 
  cbeybcf(1:ibfbc,j)=cb(1); 
  dahzxbcf(1:iefbc,j)=da(1); 
  dbhzxbcf(1:iefbc,j)=db(1); 
end 
 
%     BACK region  
 
caexbcb(1:iefbc,jbbc)=1.0; 
cbexbcb(1:iefbc,jbbc)=0.0; 
for j=2:jebc 
  y1=(j-0.5)*dx; 
  y2=(j-1.5)*dx; 
  sigmay=bcfactor*(y1^(orderbc+1)-y2^(orderbc+1)); 
  ca1=exp(-sigmay*dt/epsz); 
  cb1=(1-ca1)/(sigmay*dx); 
  caexbcb(1:iefbc,j)=ca1; 
  cbexbcb(1:iefbc,j)=cb1; 
end 
sigmay = bcfactor*(0.5*dx)^(orderbc+1); 
ca1=exp(-sigmay*dt/epsz); 
cb1=(1-ca1)/(sigmay*dx); 
caex(1:ie,jb)=ca1; 
cbex(1:ie,jb)=cb1; 
caexbcl(1:iebc,jb)=ca1; 
cbexbcl(1:iebc,jb)=cb1; 
caexbcr(1:iebc,jb)=ca1; 
cbexbcr(1:iebc,jb)=cb1; 
 
for j=1:jebc 
  y1=j*dx; 
  y2=(j-1)*dx; 
  sigmay=bcfactor*(y1^(orderbc+1)-y2^(orderbc+1)); 
  sigmays=sigmay*(muz/epsz); 
  da1=exp(-sigmays*dt/muz); 
  db1=(1-da1)/(sigmays*dx); 
  dahzybcb(1:iefbc,j)=da1; 
  dbhzybcb(1:iefbc,j)=db1; 
  caeybcb(1:ibfbc,j)=ca(1); 
  cbeybcb(1:ibfbc,j)=cb(1); 
  dahzxbcb(1:iefbc,j)=da(1); 
  dbhzxbcb(1:iefbc,j)=db(1); 
end 
 
%     LEFT region  
 
caeybcl(1,1:je)=1.0; 
cbeybcl(1,1:je)=0.0; 
for i=2:iebc 
  x1=(iebc-i+1.5)*dx; 
  x2=(iebc-i+0.5)*dx; 
  sigmax=bcfactor*(x1^(orderbc+1)-x2^(orderbc+1)); 
  ca1=exp(-sigmax*dt/epsz); 
  cb1=(1-ca1)/(sigmax*dx); 
  caeybcl(i,1:je)=ca1; 
  cbeybcl(i,1:je)=cb1; 
  caeybcf(i,1:jebc)=ca1; 
  cbeybcf(i,1:jebc)=cb1; 
  caeybcb(i,1:jebc)=ca1; 
  cbeybcb(i,1:jebc)=cb1; 
end 
sigmax=bcfactor*(0.5*dx)^(orderbc+1); 
ca1=exp(-sigmax*dt/epsz); 
cb1=(1-ca1)/(sigmax*dx); 
caey(1,1:je)=ca1; 
cbey(1,1:je)=cb1; 
caeybcf(iebc+1,1:jebc)=ca1; 
cbeybcf(iebc+1,1:jebc)=cb1; 
caeybcb(iebc+1,1:jebc)=ca1; 
cbeybcb(iebc+1,1:jebc)=cb1; 
 
for i=1:iebc 
  x1=(iebc-i+1)*dx; 
  x2=(iebc-i)*dx; 
  sigmax=bcfactor*(x1^(orderbc+1)-x2^(orderbc+1)); 
  sigmaxs=sigmax*(muz/epsz); 
  da1=exp(-sigmaxs*dt/muz); 
  db1=(1-da1)/(sigmaxs*dx); 
  dahzxbcl(i,1:je)=da1; 
  dbhzxbcl(i,1:je)=db1; 
  dahzxbcf(i,1:jebc)=da1; 
  dbhzxbcf(i,1:jebc)=db1; 
  dahzxbcb(i,1:jebc)=da1; 
  dbhzxbcb(i,1:jebc)=db1; 
  caexbcl(i,2:je)=ca(1); 
  cbexbcl(i,2:je)=cb(1); 
  dahzybcl(i,1:je)=da(1); 
  dbhzybcl(i,1:je)=db(1); 
end 
 
%     RIGHT region  
 
caeybcr(ibbc,1:je)=1.0; 
cbeybcr(ibbc,1:je)=0.0; 
for i=2:iebc 
  x1=(i-0.5)*dx; 
  x2=(i-1.5)*dx; 
  sigmax=bcfactor*(x1^(orderbc+1)-x2^(orderbc+1)); 
  ca1=exp(-sigmax*dt/epsz); 
  cb1=(1-ca1)/(sigmax*dx); 
  caeybcr(i,1:je)=ca1; 
  cbeybcr(i,1:je)=cb1; 
  caeybcf(i+iebc+ie,1:jebc)=ca1; 
  cbeybcf(i+iebc+ie,1:jebc)=cb1; 
  caeybcb(i+iebc+ie,1:jebc)=ca1; 
  cbeybcb(i+iebc+ie,1:jebc)=cb1; 
end 
sigmax=bcfactor*(0.5*dx)^(orderbc+1); 
ca1=exp(-sigmax*dt/epsz); 
cb1=(1-ca1)/(sigmax*dx); 
caey(ib,1:je)=ca1; 
cbey(ib,1:je)=cb1; 
caeybcf(iebc+ib,1:jebc)=ca1; 
cbeybcf(iebc+ib,1:jebc)=cb1; 
caeybcb(iebc+ib,1:jebc)=ca1; 
cbeybcb(iebc+ib,1:jebc)=cb1; 
 
for i=1:iebc 
  x1=i*dx; 
  x2=(i-1)*dx; 
  sigmax=bcfactor*(x1^(orderbc+1)-x2^(orderbc+1)); 
  sigmaxs=sigmax*(muz/epsz); 
  da1=exp(-sigmaxs*dt/muz); 
  db1=(1-da1)/(sigmaxs*dx); 
  dahzxbcr(i,1:je) = da1; 
  dbhzxbcr(i,1:je) = db1; 
  dahzxbcf(i+ie+iebc,1:jebc)=da1; 
  dbhzxbcf(i+ie+iebc,1:jebc)=db1; 
  dahzxbcb(i+ie+iebc,1:jebc)=da1; 
  dbhzxbcb(i+ie+iebc,1:jebc)=db1; 
  caexbcr(i,2:je)=ca(1); 
  cbexbcr(i,2:je)=cb(1); 
  dahzybcr(i,1:je)=da(1); 
  dbhzybcr(i,1:je)=db(1); 
end 

if backwall == 0;
    caexbcl(1:8,(je/2+ceil(b/(2*dx)))) = ca(2);
    cbexbcl(1:8,(je/2+ceil(b/(2*dx)))) = cb(2);
    caexbcl(1:8,(je/2-floor(b/(2*dx)))) = ca(2);
    cbexbcl(1:8,(je/2-floor(b/(2*dx)))) = cb(2);
    caexbcr(1:8,(je/2+ceil(b/(2*dx)))) = ca(2);
    cbexbcr(1:8,(je/2+ceil(b/(2*dx)))) = cb(2);
    caexbcr(1:8,(je/2-floor(b/(2*dx)))) = ca(2);
    cbexbcr(1:8,(je/2-floor(b/(2*dx)))) = cb(2);
end
    
    
%     for i = 1:iebc
%         j = length(exbcf/2+ceil(b/(2*dx));
%         caexbcr(i,j) = ca(2);
%         cbexbcr(i,j) = cb(2);
%         j = je/2-floor(b/(2*dx));
%         caexbcr(i,j) = ca(2);
%         cbexbcr(i,j) = cb(2);
%     end
% end
clear i j


if realtime == true
    figure(2)
    %set(gcf,'DoubleBuffer','on');
end

%*********************************************************************** 
%     BEGIN TIME-STEPPING LOOP 
%*********************************************************************** 
 
for n=1:nmax 
 
%*********************************************************************** 
%     Update electric fields (EX and EY) in main grid 
%*********************************************************************** 
 
ex(:,2:je)=caex(:,2:je).*ex(:,2:je)+... 
           cbex(:,2:je).*(hz(:,2:je)-hz(:,1:je-1)); 
 
ey(2:ie,:)=caey(2:ie,:).*ey(2:ie,:)+... 
           cbey(2:ie,:).*(hz(1:ie-1,:)-hz(2:ie,:)); 

if efieldsource == true
if softsource == true
    ey(is,js) = source(n) + ey(is,js);
else
    ey(is,js) = source(n);
end
end

%*********************************************************************** 
%     Update EX in PML regions 
%*********************************************************************** 
 
%     FRONT 
 
exbcf(:,2:jebc)=caexbcf(:,2:jebc).*exbcf(:,2:jebc)-...   
  cbexbcf(:,2:jebc).*(hzxbcf(:,1:jebc-1)+hzybcf(:,1:jebc-1)-... 
                      hzxbcf(:,2:jebc)-hzybcf(:,2:jebc)); 
ex(1:ie,1)=caex(1:ie,1).*ex(1:ie,1)-... 
  cbex(1:ie,1).*(hzxbcf(ibbc:iebc+ie,jebc)+... 
                hzybcf(ibbc:iebc+ie,jebc)-hz(1:ie,1)); 
  
%     BACK 
 
exbcb(:,2:jebc-1)=caexbcb(:,2:jebc-1).*exbcb(:,2:jebc-1)-... 
  cbexbcb(:,2:jebc-1).*(hzxbcb(:,1:jebc-2)+hzybcb(:,1:jebc-2)-... 
                        hzxbcb(:,2:jebc-1)-hzybcb(:,2:jebc-1)); 
ex(1:ie,jb)=caex(1:ie,jb).*ex(1:ie,jb)-... 
  cbex(1:ie,jb).*(hz(1:ie,jb-1)-hzxbcb(ibbc:iebc+ie,1)-... 
                 hzybcb(ibbc:iebc+ie,1)); 
  
%     LEFT 
 
exbcl(:,2:je)=caexbcl(:,2:je).*exbcl(:,2:je)-... 
  cbexbcl(:,2:je).*(hzxbcl(:,1:je-1)+hzybcl(:,1:je-1)-... 
                    hzxbcl(:,2:je)-hzybcl(:,2:je)); 
exbcl(:,1)=caexbcl(:,1).*exbcl(:,1)-... 
  cbexbcl(:,1).*(hzxbcf(1:iebc,jebc)+hzybcf(1:iebc,jebc)-... 
                 hzxbcl(:,1)-hzybcl(:,1)); 
exbcl(:,jb)=caexbcl(:,jb).*exbcl(:,jb)-... 
  cbexbcl(:,jb).*(hzxbcl(:,je)+hzybcl(:,je)-... 
                  hzxbcb(1:iebc,1)-hzybcb(1:iebc,1)); 
  
%     RIGHT 
 
exbcr(:,2:je)=caexbcr(:,2:je).*exbcr(:,2:je)-... 
  cbexbcr(:,2:je).*(hzxbcr(:,1:je-1)+hzybcr(:,1:je-1)-... 
                    hzxbcr(:,2:je)-hzybcr(:,2:je)); 
exbcr(:,1)=caexbcr(:,1).*exbcr(:,1)-... 
  cbexbcr(:,1).*(hzxbcf(1+iebc+ie:iefbc,jebc)+... 
                 hzybcf(1+iebc+ie:iefbc,jebc)-... 
                 hzxbcr(:,1)-hzybcr(:,1)); 
exbcr(:,jb)=caexbcr(:,jb).*exbcr(:,jb)-... 
  cbexbcr(:,jb).*(hzxbcr(:,je)+hzybcr(:,je)-... 
                  hzxbcb(1+iebc+ie:iefbc,1)-... 
                  hzybcb(1+iebc+ie:iefbc,1)); 
  
%*********************************************************************** 
%     Update EY in PML regions 
%*********************************************************************** 
 
%     FRONT 
 
eybcf(2:iefbc,:)=caeybcf(2:iefbc,:).*eybcf(2:iefbc,:)-... 
  cbeybcf(2:iefbc,:).*(hzxbcf(2:iefbc,:)+hzybcf(2:iefbc,:)-... 
                       hzxbcf(1:iefbc-1,:)-hzybcf(1:iefbc-1,:)); 
  
%     BACK 
 
eybcb(2:iefbc,:)=caeybcb(2:iefbc,:).*eybcb(2:iefbc,:)-... 
  cbeybcb(2:iefbc,:).*(hzxbcb(2:iefbc,:)+hzybcb(2:iefbc,:)-... 
                       hzxbcb(1:iefbc-1,:)-hzybcb(1:iefbc-1,:)); 
  
%     LEFT 
 
eybcl(2:iebc,:)=caeybcl(2:iebc,:).*eybcl(2:iebc,:)-... 
  cbeybcl(2:iebc,:).*(hzxbcl(2:iebc,:)+hzybcl(2:iebc,:)-... 
                      hzxbcl(1:iebc-1,:)-hzybcl(1:iebc-1,:)); 
ey(1,:)=caey(1,:).*ey(1,:)-... 
  cbey(1,:).*(hz(1,:)-hzxbcl(iebc,:)-hzybcl(iebc,:)); 
  
%     RIGHT 
 
eybcr(2:iebc,:)=caeybcr(2:iebc,:).*eybcr(2:iebc,:)-... 
  cbeybcr(2:iebc,:).*(hzxbcr(2:iebc,:)+hzybcr(2:iebc,:)-... 
                      hzxbcr(1:iebc-1,:)-hzybcr(1:iebc-1,:)); 
ey(ib,:)=caey(ib,:).*ey(ib,:)-... 
  cbey(ib,:).*(hzxbcr(1,:)+hzybcr(1,:)- hz(ie,:)); 
 
 
%*********************************************************************** 
%     Update magnetic fields (HZ) in main grid 
%*********************************************************************** 
 
hz(1:ie,1:je)=dahz(1:ie,1:je).*hz(1:ie,1:je)+...  
              dbhz(1:ie,1:je).*(ex(1:ie,2:jb)-ex(1:ie,1:je)+... 
                                ey(1:ie,1:je)-ey(2:ib,1:je)); 

if efieldsource == false
if softsource == true
    hz(is,js) = source(n) + hz(is,js);
else
    hz(is,js) = source(n);
end
end
 
%*********************************************************************** 
%     Update HZX in PML regions 
%*********************************************************************** 
 
%     FRONT 
 
hzxbcf(1:iefbc,:)=dahzxbcf(1:iefbc,:).*hzxbcf(1:iefbc,:)-... 
  dbhzxbcf(1:iefbc,:).*(eybcf(2:ibfbc,:)-eybcf(1:iefbc,:)); 
  
%     BACK 
  
hzxbcb(1:iefbc,:)=dahzxbcb(1:iefbc,:).*hzxbcb(1:iefbc,:)-... 
  dbhzxbcb(1:iefbc,:).*(eybcb(2:ibfbc,:)-eybcb(1:iefbc,:)); 
  
%     LEFT 
  
hzxbcl(1:iebc-1,:)=dahzxbcl(1:iebc-1,:).*hzxbcl(1:iebc-1,:)-... 
  dbhzxbcl(1:iebc-1,:).*(eybcl(2:iebc,:)-eybcl(1:iebc-1,:)); 
hzxbcl(iebc,:)=dahzxbcl(iebc,:).*hzxbcl(iebc,:)-... 
  dbhzxbcl(iebc,:).*(ey(1,:)-eybcl(iebc,:)); 
  
%     RIGHT 
  
hzxbcr(2:iebc,:)=dahzxbcr(2:iebc,:).*hzxbcr(2:iebc,:)-... 
  dbhzxbcr(2:iebc,:).*(eybcr(3:ibbc,:)-eybcr(2:iebc,:)); 
hzxbcr(1,:)=dahzxbcr(1,:).*hzxbcr(1,:)-... 
  dbhzxbcr(1,:).*(eybcr(2,:)-ey(ib,:)); 
  
%*********************************************************************** 
%     Update HZY in PML regions 
%*********************************************************************** 
 
%     FRONT 
  
hzybcf(:,1:jebc-1)=dahzybcf(:,1:jebc-1).*hzybcf(:,1:jebc-1)-... 
  dbhzybcf(:,1:jebc-1).*(exbcf(:,1:jebc-1)-exbcf(:,2:jebc)); 
hzybcf(1:iebc,jebc)=dahzybcf(1:iebc,jebc).*hzybcf(1:iebc,jebc)-... 
  dbhzybcf(1:iebc,jebc).*(exbcf(1:iebc,jebc)-exbcl(1:iebc,1)); 
hzybcf(iebc+1:iebc+ie,jebc)=... 
  dahzybcf(iebc+1:iebc+ie,jebc).*hzybcf(iebc+1:iebc+ie,jebc)-... 
  dbhzybcf(iebc+1:iebc+ie,jebc).*(exbcf(iebc+1:iebc+ie,jebc)-... 
                                  ex(1:ie,1)); 
hzybcf(iebc+ie+1:iefbc,jebc)=... 
  dahzybcf(iebc+ie+1:iefbc,jebc).*hzybcf(iebc+ie+1:iefbc,jebc)-... 
  dbhzybcf(iebc+ie+1:iefbc,jebc).*(exbcf(iebc+ie+1:iefbc,jebc)-... 
                                   exbcr(1:iebc,1)); 
 
%     BACK 
  
hzybcb(1:iefbc,2:jebc)=dahzybcb(1:iefbc,2:jebc).*hzybcb(1:iefbc,2:jebc)-... 
  dbhzybcb(1:iefbc,2:jebc).*(exbcb(1:iefbc,2:jebc)-exbcb(1:iefbc,3:jbbc)); 
hzybcb(1:iebc,1)=dahzybcb(1:iebc,1).*hzybcb(1:iebc,1)-... 
  dbhzybcb(1:iebc,1).*(exbcl(1:iebc,jb)-exbcb(1:iebc,2)); 
hzybcb(iebc+1:iebc+ie,1)=... 
  dahzybcb(iebc+1:iebc+ie,1).*hzybcb(iebc+1:iebc+ie,1)-... 
  dbhzybcb(iebc+1:iebc+ie,1).*(ex(1:ie,jb)-exbcb(iebc+1:iebc+ie,2)); 
hzybcb(iebc+ie+1:iefbc,1)=... 
  dahzybcb(iebc+ie+1:iefbc,1).*hzybcb(iebc+ie+1:iefbc,1)-... 
  dbhzybcb(iebc+ie+1:iefbc,1).*(exbcr(1:iebc,jb)-... 
                                exbcb(iebc+ie+1:iefbc,2)); 
  
%     LEFT 
  
hzybcl(:,1:je)=dahzybcl(:,1:je).*hzybcl(:,1:je)-... 
  dbhzybcl(:,1:je).*(exbcl(:,1:je)-exbcl(:,2:jb)); 
  
%     RIGHT 
  
hzybcr(:,1:je)=dahzybcr(:,1:je).*hzybcr(:,1:je)-... 
  dbhzybcr(:,1:je).*(exbcr(:,1:je)-exbcr(:,2:jb)); 
 
%*********************************************************************** 
%     Store fields in time 
%*********************************************************************** 

sink(n) = source(n);
%Note: top row physically stored at bottom(end) of array
%Note: array is stored XxY
%sink(n) = hz(is,ajcenter)';
%f1126(n) = hz(xa2num,ajcenter)';
%sink3(n) = hz(ie-fftbuffer,je/2); %#ok<SAGROW>
if twoantenna == true
    sink4(n) = mean(ey(iread,jread));
else
    sink4(n) = ey(iread,jread);
    sink5(n) = ey(iread2,jread2);
    sink6(n) = ey(iread2,jread2+5*scale);
    sink7(n) = ey(iread2,jread2-5*scale);
    sink8(n) = ey(iread2,jread2+10*scale);
    sink9(n) = ey(iread2,jread2-10*scale);
    sink10(n) = ey(iread2,jread2+15*scale);
    sink11(n) = ey(iread2,jread2-15*scale);
    sink12(n) = ey(iread2,jread2+20*scale);
    sink13(n) = ey(iread2,jread2-20*scale);
    sink14(n) = ey(iread2,jread2+25*scale);
    sink15(n) = ey(iread2,jread2-25*scale);
    sink16(n) = ey(iread2,jread2+30*scale);
    sink17(n) = ey(iread2,jread2-30*scale);
    sink18(n) = ey(iread2,jread2+35*scale);
    sink19(n) = ey(iread2,jread2-35*scale);
    sink20(n) = ey(iread2,jread2+40*scale);
    sink21(n) = ey(iread2,jread2-40*scale);
end
% if n == 800
%    sink3 = hz(60:160,xa2num)';
% end

% Store Fields for nff

if ss == true
    
    % Note: A sinusoidal source is needed
    if n == n1
    %exb = ex(fftbuffer:ie-fftbuffer,je-fftbuffer)';
    %eyl = ey(fftbuffer,je/2:je-fftbuffer);
    eyr = ey(ie-fftbuffer,je/2:je-fftbuffer);
    %hzb = hz(fftbuffer:ie-fftbuffer,je-fftbuffer)';
    %hzl = hz(fftbuffer,je/2:je-fftbuffer);
    hzr = hz(ie-fftbuffer,je/2:je-fftbuffer);
    end
    if n == n2
    %exb = exb+1i*ex(fftbuffer:ie-fftbuffer,je-fftbuffer)';
    %eyl = eyl+1i*ey(fftbuffer,je/2:je-fftbuffer);
    eyr = eyr+1i*ey(ie-fftbuffer,je/2:je-fftbuffer);
    %hzb = hzb+1i*hz(fftbuffer:ie-fftbuffer,je-fftbuffer)';
    %hzl = hzl+1i*hz(fftbuffer,je/2:je-fftbuffer);
    hzr = hzr+1i*hz(ie-fftbuffer,je/2:je-fftbuffer);
    end
end


%*********************************************************************** 
%     Visualize fields 
%*********************************************************************** 
 
if realtime == true
    timeplot(ex,ey,hz,ib,jb,ie,je,n);
end

%*********************************************************************** 
%     END TIME-STEPPING LOOP 
%*********************************************************************** 
 
end 
 
% movie(gcf,M,0,10,rect);

%*********************************************************************** 
%     Near to Far-Field Transformation 
%*********************************************************************** 

if nff == true
if ss == true %Sinusoidal
    [hzrfft phi] = spacialfft(-2*muz*hzr,dx,N,64);
    eyrfft = spacialfft(-2*epsz*eyr,dx,N,64);
    ep = -1i*omega*(hzrfft.*cos(phi) + etaz*eyrfft);%-hzbfft*sin(phib)
end
end

%*********************************************************************** 
%     Visualization 
%*********************************************************************** 

% f1122 = fft(sink3);
% figure(4) %Figure 11-22
% subplot(2,1,1)
% plot(abs(f1122))
% subplot(2,1,2)
% plot(angle(f1122)*180/pi)

% Figure 11-26
% n = 300:1500;
% plot(n,f1126(n))

% NFF plots
if ss == true
    % [hzrfft phi] = postfdtd(-2*muz*hzr,B,N);
    figure(5)
    amp = 20*log10(abs(ep));
    degree = phi*180/pi;
    plot(degree,amp)
    ylabel('dB'),xlabel('\phi \circ')
end

% if batchfile == true
%     n = 1:nmax;
%     figure(run)
%     subplot(2,1,1)
%     plot(n,sink5)
%     subplot(2,1,2)
%     plot(n,sink4)
%     source = sink;
%     output = sink4;
%     figure(2)
%     subplot(2,1,1)
%     plot(n,sink6)
%     subplot(2,1,2)
%     plot(n,sink7)
%     figure(3)
%     subplot(2,1,1)
%     plot(n,sink8)
%     subplot(2,1,2)
%     plot(n,sink9)
%     figure(4)
%     subplot(2,1,1)
%     plot(n,sink10)
%     subplot(2,1,2)
%     plot(n,sink11)
%     figure(5)
%     subplot(2,1,1)
%     plot(n,sink12)
%     subplot(2,1,2)
%     plot(n,sink13)
%     figure(6)
%     subplot(2,1,1)
%     plot(n,sink14)
%     subplot(2,1,2)
%     plot(n,sink15)
%      figure(7)
%     subplot(2,1,1)
%     plot(n,sink16)
%     subplot(2,1,2)
%     plot(n,sink17)
%     figure(8)
%     subplot(2,1,1)
%     plot(n,sink18)
%     subplot(2,1,2)
%     plot(n,sink19)
%     figure(9)
%     subplot(2,1,1)
%     plot(n,sink20)
%     subplot(2,1,2)
%     plot(n,sink21)
% end

%     figure(5)
%     a = 20*log10(abs(fieldfft));
%     b = phi*180/pi;
%     plot(b,a)
%     ylabel('dB'),xlabel('\phi \circ')
if scale == 1
    save myfdtddata20.mat
elseif scale == 2
    save myfdtddata40.mat
end
runtime = toc;
runtime = num2str(runtime);
disp( ['it takes ' runtime ' seconds'] )
'''