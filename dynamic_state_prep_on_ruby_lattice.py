###
### This code was written by Ruben Verresen.
### If you use/adapt this code, please cite the corresponding github repository of user 'rubenverr', as well as arxiv:2104.04119.
### Since this code uses the tenpy library, please also cite Johannes Hauschild, Frank Pollmann, SciPost Phys. Lect. Notes 5 (2018).
###
### This file dynamically prepares a state by changing the Hamiltonian as a fucntion of time (initial state is empty), mimicking the experimental set-up
###

from tenpy.networks.mps import MPS
import numpy as np
from tenpy.models.rydberg_on_ruby import ruby_rydberg
from tenpy.algorithms import dmrg, mpo_evolution
from tenpy.networks.mpo import MPOEnvironment
import time as time
import pylab as plt
import sys

chi = 100
bc = 'infinite'
Rb = 2.4
Omega = 1.
imag = False
dt = 0.01
Ly = 4
nring = 1
random = False

radius = 2.8
save = True
savename = 'testest'
save_every = 9 #99

factor = 1

Ly_save = int(Ly)
def model(delta,Omega):
	M = ruby_rydberg({"bc_MPS":bc,"Rb":Rb,"Omega":Omega,"radius":radius,"delta":delta, "Ly":Ly_save,"nring":nring})
	return M

M = model(0,1)

print( "MPO bond dimensions =", M.H_MPO.chi )
#print(M.lat.order)

def number(x):
	if x=='A': return 0
	elif x=='B': return 1
	elif x=='C': return 2

def letter(x):
	if x==1: return 'A'
	elif x==2: return 'B'
	elif x==3: return 'C'

def get_opstring(x,stringtype):
	if x==0: return 'P2 P3' if stringtype=='P' else 'Q1'
	elif x==1: return 'P1 P3' if stringtype=='P' else 'Q2'
	elif x==2: return 'P1 P2' if stringtype=='P' else 'Q3'

def get_string(phi,indices,stringtype): ### indices being a list of strings, each string of the form '4B' where 4 is triangle index and B is sublattice
	r_list = np.array([int(index[:-1]) for index in indices])
	d_list = np.array([number(index[-1]) for index in indices])
	d_list = d_list[np.argsort(r_list)]
	r_list = np.sort(r_list)
	op_list = ['Id']*r_list[0] + [get_opstring(d_list[0],stringtype)]
	for i in range(1,r_list.shape[0]): op_list += ['Id']*( r_list[i]-r_list[i-1]-1 ) + [get_opstring(d_list[i],stringtype)]
	return op_list

def corr(phi,indices,stringtype,i0=0):
	op_list = get_string(psi,indices,stringtype)
	return phi.expectation_value_multi_sites(op_list,i0=i0)[()]

Pcirc = [x for i in range((Ly+2)//4) for x in [str(2*i)+'C',str(2*i+1)+'B']] #ceil(Ly/4-1e-10)
Qcirc = Pcirc + [x for i in range(Ly//2) for x in [str(2*i)+'B',str(2*i+1)+'A']][-2*(Ly//4):]

def shift(oplist,n): return [str(int(x[:-1])+n)+x[-1:] for x in oplist]

def density(phi):
	n1 = phi.expectation_value('n1')
	n2 = phi.expectation_value('n2')
	n3 = phi.expectation_value('n3')
	n = []
	for i in range(psi.L): n += [n1[i],n2[i],n3[i]]
	return np.array(n)

psi = MPS.from_product_state(M.lat.mps_sites(), [0]*Ly, bc)

par = {
	'N_steps':1, #10,
	'compression_method': 'variational', #'SVD' | 'variational'
	'trunc_params': {"chi_max": chi,"svd_min": 1e-10,},
}

from scipy.interpolate import interp1d
def splineSweep(tTotal, tInf, fMin, fMax, fInf, slope,midT=0.05):
	midF = slope * midT
	tPoints1 = np.array([0., tInf-midT, tInf, tInf+midT, tTotal])
	fPoints1 = np.array([fMin, fInf-midF, fInf, fInf+midF, fMax])
	fInterp1 = interp1d(tPoints1, fPoints1, kind='cubic',
	bounds_error=False, fill_value=(fMin, fMax))
	return fInterp1

T = 2*(2*np.pi*1.4)

steps = int(T/dt)
steps = int(steps/par['N_steps'])
dt = T/(par['N_steps']*steps)
par['dt'] = dt
steps *= factor
print("time-evolving until t =", T*factor, "with", steps*par['N_steps'], "sweeps using dt =", dt)

t = np.arange(0,T,dt/factor)
Omega = 1.4

sweep = splineSweep(tTotal = T/(2*np.pi*Omega), tInf = 1, fMin = -8, fMax = 9.4, fInf = 2.4, slope = 2)
f_delta = sweep(t/(2*np.pi*Omega))/Omega

f_Omega = f_delta*0 + 1

#f_t = t*factor

a = f_delta.shape[0]

temp = np.array([f_delta[0]]*(steps//4))
f_delta = np.concatenate([temp,f_delta])
temp = np.linspace(0,1,steps//4)
f_Omega = np.concatenate([temp,f_Omega])
steps += steps//4

print(psi.chi)
L = Ly

sweeps = []
xi_list = [psi.correlation_length()]
S_list = [psi.entanglement_entropy()[0]]
delta_list = [f_delta[0]]
Omega_list = [f_Omega[0]]
temp = ['P1 P2']+['P1 P3']+['Id']*3+['P1 P2']+['P2 P3']+['Id']+['P1 P3']+['P2 P3']
star_list = [psi.expectation_value_multi_sites(temp,i0=4+2*Ly*1)[()]]
n_list = [np.mean(density(psi))]
E_list = [M.H_MPO.expectation_value(psi)/3]
Ploop_list = [corr(psi,Pcirc,'P')]
Ploopcorr_list = [ [corr(psi,Pcirc+shift(Pcirc,Ly*i),'P') for i in range(10)] ]
Qloop_list = [corr(psi,Qcirc,'Q')]
Qnum_list = [[0]*(Ly//2-1)]
Pnum_list = [[0]*(Ly//2-1)]
Qden_list = [[0]*(Ly//2-1)]
Pden_list = [[0]*(Ly//2-1)]
VBS_list = [ [psi.correlation_function('n1','n1',sites1=[0],sites2=[Ly*i])[0,0] - np.mean(density(psi))**2 for i in range(1,10)] ]

prob_monomer_list = [1]
prob_dimer_list = [0]
prob_doubledimer_list = [0]
Pstring_exp_list = [1]
Qstring_exp_list = [0]

t_list = [0]
t0 = time.time()
for i in range(steps-1):
	counter = i
	delta = f_delta[i+1]
	Omega = f_Omega[i+1]
	delta_list.append( delta )
	Omega_list.append( Omega )
	M = model(delta,Omega)
	evol = mpo_evolution.ExpMPOEvolution(psi,M,par)
	evol.run()
	psi.canonical_form()
	print("\ncompleted", (i+1)*par['N_steps'], "sweeps")
	print("delta =", delta)
	print("Omega =", Omega)
	print("Omega*t =", par['dt']*(i+1)*par['N_steps'] )
	print("took", (time.time()-t0)/60., "minutes so far")
	t_list.append( par['dt']*(i+1)*par['N_steps'] )
	print("chi =", psi.chi)
	xi = psi.correlation_length()
	S = psi.entanglement_entropy()[0]
	E = M.H_MPO.expectation_value(psi)/3
	n = density(psi)
	print( "n =", np.mean(n) )
	print( "<n> =", n )
	print("xi =", xi )
	print( "S between rings =", S )
	S_list.append( S )
	xi_list.append( xi )
	n_list.append( np.mean(n) )
	E_list.append( E )

	Ploop = corr(psi,Pcirc,'P')
	Ploopcorr = [corr(psi,Pcirc+shift(Pcirc,Ly*i),'P') for i in range(10)]
	Qloop = corr(psi,Qcirc,'Q')
	print("Ploop =", Ploop)
	print("Ploopcorr =", Ploopcorr)
	print("Qloop = ", Qloop)
	Ploop_list.append(Ploop)
	Qloop_list.append(Qloop)
	Ploopcorr_list.append(Ploopcorr)

	if 1:
		monomer = psi.correlation_function('proj2 proj3','proj1 proj2',sites1=[0],sites2=[1])[0,0]
		print( "probability for monomer =", monomer )

		dimer_list = [('n2 proj3','proj1 proj2'),('proj2 n3','proj1 proj2'),('proj2 proj3','n1 proj2'),('proj2 proj3','proj1 n2')]
		dimer = 0
		for blah in dimer_list: dimer += psi.correlation_function(blah[0],blah[1],sites1=[0],sites2=[1])[0,0]
		print( "probability for dimer =", dimer )

		doubledimer_list = [('n2 proj3','n1 proj2'),('n2 proj3','proj1 n2'),('proj2 n3','n1 proj2'),('proj2 n3','proj1 n2')]
		doubledimer = 0
		for blah in doubledimer_list: doubledimer += psi.correlation_function(blah[0],blah[1],sites1=[0],sites2=[1])[0,0]
		print( "probability for double-dimer =", doubledimer )

		print( "consistency check: 4*n - (dimer+2*doubledimer) =", 4*np.mean(density(psi)) - (dimer+2*doubledimer) )

		prob_monomer_list.append( monomer )
		prob_dimer_list.append( dimer )
		prob_doubledimer_list.append( doubledimer )
	
	I = []
	S = []
	V = []

	I += [[str(Ly//2)+'C',str(Ly+1)+'C',str(Ly+Ly//2)+'B']]
	S += ['P']
	Pstring_exp_list.append( corr(psi,I[-1],S[-1]) )
	print("half star parity =", Pstring_exp_list[-1])
	I += [[str(Ly//2)+'C',str(Ly+1)+'C',str(Ly+Ly//2)+'B']]
	S += ['Q']
	Qstring_exp_list.append( corr(psi,I[-1],S[-1]) )
	print("half hexagon resonance =", Qstring_exp_list[-1])

	Q_num = [] ### numerator for BFFM
	Q_den = [] ### denominator for BFFM
	### NxN Q-BFFM
	for N in range(1,Ly//2):
		temp = [X for i in range(N) for X in [str(L//2+i*L)+'A',str(L+1+i*L)+'C']]
		temp += [X for i in range(N) for X in [str(N*L+L//2+(L//2)*i+2*((i+1)//2))+'B',str(N*L+L//2+1+(L//2)*i+2*((i+1)//2))+'B']]
		I += [list(temp)]
		S += ['Q']

		temp[0] = temp[0][:-1]+'C'; temp[-1] = temp[-1][:-1]+'A'
		temp += [X for i in range(N) for X in [str(L//2+(L//2)*i+2*((i+1)//2))+'B',str(L//2+1+(L//2)*i+2*((i+1)//2))+'B']][1:]
		temp += [X for i in range(N) for X in [str(i*L+L//2+(L//2)*N+(2*((N+1)//2))%(L//2) )+'A',str((i+1)*L+L//2+1+(L//2)*(N-1)+(2*(N//2))%(L//2) )+'C']][:-1]
		I += [list(temp)]
		S += ['Q']

		Q_num.append(corr(psi,I[-2],S[-2]))
		Q_den.append(corr(psi,I[-1],S[-1]))
		print("Q_BFFM for "+str(N)+"x"+str(N)+" =", corr(psi,I[-2],S[-2])/np.sqrt(np.abs(corr(psi,I[-1],S[-1]))) )

	P_num = [] ### numerator for BFFM
	P_den = [] ### denominator for BFFM
	### NxN P-BFFM
	for N in range(1,Ly//2):
		temp = [str(L+1+i*L)+'C' for i in range(N)]
		temp += [str(N*L+L//2+(L//2)*i+2*((i+1)//2))+'B' for i in range(N)]
		I += [list(temp)]
		S += ['P']

		temp += [str(L//2+1+(L//2)*i+2*((i+1)//2))+'B' for i in range(N)]
		temp += [X for i in range(N) for X in [str(i*L+L//2+(L//2)*N+(2*((N+1)//2))%(L//2) )+'A']]
		temp += [str(L//2)+'C',str(N*L+1+L//2+(L//2)*(N-1)+2*(N//2))+'A']
		I += [list(temp)]
		S += ['P']

		P_num.append(corr(psi,I[-2],S[-2]))
		P_den.append(corr(psi,I[-1],S[-1]))
		print("P_BFFM for "+str(N)+"x"+str(N)+" =", corr(psi,I[-2],S[-2])/np.sqrt(np.abs(corr(psi,I[-1],S[-1]))) )

	print( "Q_num =", Q_num )
	print( "Q_den =", Q_den )
	print( "P_num =", P_num )
	print( "P_den =", P_den )
	Qnum = Qnum_list.append( Q_num )
	Qden = Qden_list.append( Q_den )
	Pnum = Pnum_list.append( P_num )
	Pden = Pden_list.append( P_den )

	VBS = [psi.correlation_function('n1','n1',sites1=[0],sites2=[Ly*i])[0,0] - np.mean(density(psi))**2 for i in range(1,10)]
	VBS_list.append(VBS)
	print( "<nn> - <n>^2 =", VBS)

	print( "t_list =", t_list[-1] )

	res = {
		't' : t_list,
		'delta' : delta_list,
		'Omega': Omega_list,
		'n' : n_list,
		'S' : S_list,
		'xi' : xi_list,
		'E' : E_list,
		'Ploop' : Ploop_list,
		'Qloop' : Qloop_list,
		'Ploopcorr' : Ploopcorr_list,
		'P_num' : Pnum_list,
		'Q_num' : Qnum_list,
		'P_den' : Pden_list,
		'Q_den' : Qden_list,
		'VBS' : VBS_list,
		'monomer' : prob_monomer_list,
		'dimer' : prob_dimer_list,
		'doubledimer' : prob_doubledimer_list,
		'Pstring_exp' : Pstring_exp_list,
		'Qstring_exp' : Qstring_exp_list,
	}

	if save and counter%save_every==0:
		temp = psi.copy()
		temp.canonical_form()

		folder = ''
		np.save(folder+'state.npy',temp._B)
		np.save(folder+'res.npy',res)



