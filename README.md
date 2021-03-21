
## For Nbody simulation with fields
This project is still going on....

This project aims at N-body simulations which can incorporate dynamics of quintessence dark energy models by solving relativistic field equations. So along with particles, we solve for metric field(represented by two Bardeen Potentials) and scalar field(quintessence). The aim is to have a code which can have nonlinear source term in both matter and scalar field.

Current Status: Dynamics module seems to work. A comprehensive initial condition generator needs to be developed. 


Basic equations are provided in this [document](https://github.com/manu0x/nbody/blob/master/equations.pdf).
This repository contains programs with varying level of approximations. Current working branch is using a mix of finite 
difference and Fourier transform based tehcniques.
All C codes here are N-body codes. These are at varying level of approximations and may use different numerical scheme.
This project incudes different levels/choices of approximations:
- All equations(in all codes here) have only upto first order terms in metric. 
- Allowing nonlinear quintessence field or keeping it upto linear order. Codes that keep only linear terms in quintessence field have "linear" in their names.
- Keeping or dropping relativistic equations for particle motions. All codes that have "nr" in their names use non-relativistic equation of motion for particles.

Further some codes here use explicit integration scheme while others use implicit scheme. This system of equations shows numerical instability at small scales.
This can be taken care of by using implicit numerical methods. But implicit methods pose big challenge as inversion of nonlinear equations is computationally
very challenging for large no. of points, which is the case here. Here we try a simple partially implicit scheme, which seems to work. To get an idea about the terms 
causing numerical instability and the trick that resolves it please have a look at the last few slides of this [talk](http://www.icehap.chiba-u.jp/IAU_B1/chaica2020/video/CaICA-II%20Nov.21/Nov21_06_Rajvanshi.mp4). 

All codes using implicit mathods have "implicit" in their names. For example: code [code1exp_nr_p_implicit.c](https://github.com/manu0x/nbody/blob/minimal_finit_d_no_ft/code1exp_nr_p_implicit.c) 
has only non-relativistic particles and uses implicit methods while [code1exp_rel_p_implicit.c](https://github.com/manu0x/nbody/blob/minimal_finit_d_no_ft/code1exp_rel_p_implicit.c)
uses relativistic equations of motion for particles. 
