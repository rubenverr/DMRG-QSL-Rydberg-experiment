###
### This code was written by Ruben Verresen.
### If you use/adapt this code, please cite the corresponding github repository of user 'rubenverr', as well as arxiv:2104.04119.
### Since this code uses the tenpy library, please also cite Johannes Hauschild, Frank Pollmann, SciPost Phys. Lect. Notes 5 (2018).
###
### This file creates the model for Rydberg atoms on the ruby lattice.
###

import numpy as np
from .lattice import Site, Chain, Lattice
from .model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel, MultiCouplingModel
from ..linalg import np_conserved as npc
from ..tools.params import get_parameter, unused_parameters
from ..networks.site import SpinSite
from math import ceil

__all__ = ['ruby_rydberg']

class ruby_rydberg(CouplingModel,MPOModel):
    def __init__(self, model_params):
        name = "ruby rydberg"
        self.Ly = Ly = model_params.get('Ly',4)
        self.nring = nring = model_params.get('nring', 1)
        Rb = model_params.get('Rb', 1.)
        radius = model_params.get('radius',Rb)
        Omega = model_params.get('Omega',0)
        delta = model_params.get('delta',0)
        phase = model_params.get('phase', 0); print("phase =", phase)
        different_order = model_params.get('different_order',False)
        bc_MPS = model_params.get( 'bc_MPS', 'finite')
        unused_parameters(model_params, name)
        leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1]),[0]*4)
        n_sum = np.diag([0,1,1,1])
        n1 = np.diag([0,1,0,0])
        n2 = np.diag([0,0,1,0])
        n3 = np.diag([0,0,0,1])
        P1 = np.diag([1,-1,1,1])
        P2 = np.diag([1,1,-1,1])
        P3 = np.diag([1,1,1,-1])
        proj1 = np.diag([1,0,1,1])
        proj2 = np.diag([1,1,0,1])
        proj3 = np.diag([1,1,1,0])
        X_sum = np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
        X1 = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
        X2 = np.array([[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]])
        X3 = np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]])
        Y1 = np.array([[0,-1j,0,0],[1j,0,0,0],[0,0,0,0],[0,0,0,0]])
        Y2 = np.array([[0,0,-1j,0],[0,0,0,0],[1j,0,0,0],[0,0,0,0]])
        Y3 = np.array([[0,0,0,-1j],[0,0,0,0],[0,0,0,0],[1j,0,0,0]])
        Q1 = np.zeros((4,4)); Q2 = np.zeros((4,4)); Q3 = np.zeros((4,4))
        sign = -np.sign(Omega)
        if np.abs(sign)<1e-10: sign = 1
        print("sign =", sign)
        Q1[0,1] = sign; Q1[2,3] = 1; Q1 += Q1.T
        Q2[0,2] = sign; Q2[1,3] = 1; Q2 += Q2.T
        Q3[0,3] = sign; Q3[1,2] = 1; Q3 += Q3.T
        phi = np.pi/2; A = np.diag([1,np.exp(1j*phi),np.exp(1j*phi),1])
        phi = 0; B = np.diag([1,1,np.exp(1j*phi),np.exp(1j*phi)])
        phi = -np.pi/2; C = np.diag([1,np.exp(1j*phi),np.exp(1j*phi),1])
        phi = 0; D = np.diag([1,1,np.exp(1j*phi),np.exp(1j*phi)])
        site = Site(leg,['0','1','2','3'],n_sum=n_sum,n1=n1,n2=n2,n3=n3,X_sum=X_sum,X1=X1,X2=X2,X3=X3,Y1=Y1,Y2=Y2,Y3=Y3,P1=P1,P2=P2,P3=P3,Q1=Q1,Q2=Q2,Q3=Q3,A=A,B=B,C=C,D=D,proj1=proj1,proj2=proj2,proj3=proj3)
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
        self.lat = lat = Lattice([nring],[site]*Ly,bc=bc,bc_MPS=bc_MPS,order='default')
        temp = [0,1,2,3]
        order = []
        self.s = s = np.sqrt(3) + 1/np.sqrt(3)
        Lcirc = Ly*3*s/4
        self.N_uc = N_uc = int(ceil(Ly/4))
        self.uc = uc = 4
        for i in range(N_uc): order += [temp[0]+i*4,temp[1]+i*4]
        for i in range(N_uc): order += [temp[2]+i*4,temp[3]+i*4]
        if Ly%4==2: order = order[:-2]
        self.order = order
        temp = np.zeros((nring*Ly,2))
        for i in range(nring):
                temp[Ly*i:Ly*(i+1),0] = i
                temp[Ly*i:Ly*(i+1),1] = np.array(order)
        lat.order = temp.astype(int)
        CouplingModel.__init__(self, lat)

        def coor(a,b,c,d,center=False): return self.coor(a,b,c,d,center=center)

        lw = 1; mw = 5
        r_list = []
        a1 = 0
        num = 0
        for b1 in range(N_uc):
            if Ly%4==2 and b1==N_uc-1: num_of_sites = range(uc-2)
            else: num_of_sites = range(uc)
            for c1 in num_of_sites:
                for d1 in range(3):
                    r1 = coor(a1,b1,c1,d1)
                    a2 = 0; non_zero_couplings = True
                    while non_zero_couplings or a2<a1+2:
                        non_zero_couplings = False
                        for b2 in range(N_uc):
                            if Ly%4==2 and b2==N_uc-1: num_of_sites = range(uc-2)
                            else: num_of_sites = range(uc)
                            for c2 in num_of_sites:
                                for d2 in range(3):
                                    r = np.min([ np.sqrt(np.sum( (r1-coor(a2,b2+i*N_uc,c2,d2))**2 )) for i in [-1,0,1]])
                                    if r<radius+1e-10 and np.abs(a1-a2)+np.abs(b1-b2)+np.abs(c1-c2)>1e-10:
                                        r_list.append(r)
                                        if a2>a1: factor = 1.
                                        else: factor = 0.5
                                        non_zero_couplings = True
                                        coupling = factor*np.abs(Omega)/(r/Rb)**6
                                        self.add_coupling(coupling,c1+b1*uc,'n'+str(d1+1),c2+b2*uc,'n'+str(d2+1),a2-a1)
                                        num += int( 2*factor + 1e-10 )
                        a2 += 1
        print("on average, every site is coupled to", 2+num/(3*Ly), "sites")
        rmax = np.max(r_list)
        print("\nr_max =", rmax, ", L_circ =", Lcirc, ", r_max / L_circ =", rmax/Lcirc)
        r_list = [int(np.round(r*1e10)) for r in r_list]
        r_list = list(dict.fromkeys(r_list))
        r_list = np.sort( np.array(r_list)*1e-10 )
        print("all radii that were included =", r_list, "\n")

        for i in range(Ly):
            self.add_onsite(-delta,i,'n_sum')
            self.add_onsite(Omega/2,i,'X_sum')

        MPOModel.__init__(self, lat, self.calc_H_MPO())

    def coor(self,a,b,c,d,center=False):
        temp = 1./(2*np.sqrt(3))
        bc_type = 0
        s = np.sqrt(3) + 1/np.sqrt(3)
        x = a*np.sqrt(3)*s
        y = b*3*s
        if c==(0-bc_type)%4:
            x += 0
            y += (-1+3*bc_type)*s
        elif c==1-bc_type:
            x += 0; y += 0
        elif c==2-bc_type:
            x += s*np.cos(np.pi/6)
            y += s*np.sin(np.pi/6)
        elif c==3-bc_type:
            x += s*np.cos(np.pi/6)
            y += s*np.sin(np.pi/6) + s
        if c%2==1-bc_type and not center:
            if d==0: x += -0.5; y += -temp
            elif d==1: x += 0.5; y += -temp
            elif d==2: y += np.sqrt(3)/2 - temp
        elif not center:
            if d==2: x -= -0.5; y -= -temp
            elif d==1: x -= 0.5; y -= -temp
            elif d==0: y -= np.sqrt(3)/2 - temp
        return np.array([x,y])

