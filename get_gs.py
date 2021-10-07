###
### This code was written by Ruben Verresen.
### If you use/adapt this code, please cite the corresponding github repository of user 'rubenverr', as well as arxiv:2104.04119.
### Since this code uses the tenpy library, please also cite Johannes Hauschild, Frank Pollmann, SciPost Phys. Lect. Notes 5 (2018).
###
### This file obtains the ground state using DMRG and calculates various quantities.
###

from tenpy.networks.mps import MPS
import numpy as np
from tenpy.models.rydberg_on_ruby import ruby_rydberg
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPOEnvironment
from math import ceil
import tenpy
tenpy.show_config()
import logging
logging.basicConfig(level=logging.INFO)

chi_list = [20, 50, 100, 200, 400, 700, 1000]
bc = 'infinite'
delta = 1.6
Omega = 1
Rb = 2.4
radius = 2.8
Ly = 4
nring = 2

random = True
different_order = False
save = False
savename = 'test'
random = False
save_res = False
max_sweeps = 100
min_sweeps = 20

M = ruby_rydberg({"bc_MPS":bc,"Omega":Omega,"Rb":Rb,"radius":radius,"delta":delta,"Ly":Ly,"nring":nring,"different_order":different_order})

if random: startwf = [np.random.choice([0,1]) for i in range(int(Ly)*nring)]
else: startwf = [0,1]*int(Ly/2)*nring

print( "MPO bond dimensions =", M.H_MPO.chi )
#print(M.lat.order)

def number(x):
	if x=='A': return 0
	elif x=='B': return 1
	elif x=='C': return 2

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

psi = MPS.from_product_state(M.lat.mps_sites(), startwf, bc)
psi.canonical_form()
for i in range(nring): print( "parity loop around circumference =", corr(psi,Pcirc,'P',i0=Ly*i) )

dmrg_params = {"trunc_params": {"chi_max": chi_list[0],"svd_min": 1e-10,},
	'max_sweeps': max_sweeps, #100, #300,
	'min_sweeps': min_sweeps,
	'max_E_err': 1e-9,
	#'max_E_err': 1e-10,
	#'max_S_err': 1e-5,
	#'start_env': 100,
	'mixer' : True
}

eng = dmrg.TwoSiteDMRGEngine(psi,M,dmrg_params)

S_list = []
xi_list = []
Sring_list = []
n_list = []
n2_list = []
E_list = []
Pstar_list = []
P2star_list = []
P3star_list = []
Ploop_list = []
Pstring_list = []

Qloop_list = []
Qhexagon_list = []
Q2hexagon_list = []
Q3hexagon_list = []

Qstring_list = []
FMdiag_list = []
FMoffdiag_list = []

Ploopcorr_list = []
Qnum_list = []
Pnum_list = []
Qden_list = []
Pden_list = []
VBS_list = []

for i in range(len(chi_list)):
	chi = chi_list[i]
	print( "chi =", chi )

	eng.reset_stats()
	eng.trunc_params['chi_max'] = chi

	eng.run()
	eng.engine_params['mixer'] = None
	eng.engine_params.pop('start_env',None)
	E0 = M.H_MPO.expectation_value(psi)
	E0 = E0/3

	folder = ''
	if save:
		temp = psi.copy()
		temp.canonical_form()
		np.save(folder+'psi.npy',temp._B)
		del temp

	if bc=='infinite':
		xi = psi.correlation_length()
		xi = xi*(3+1)/Ly
		xi_list.append( xi )
		Sring_list.append( psi.entanglement_entropy()[0] )
		temp = [corr(psi,Pcirc,'P',i0=Ly*i0) for i0 in range(nring)]
		Ploop_list.append( temp )
		print( "P-loop =", Ploop_list  )
		temp = [corr(psi,Qcirc,'Q',i0=Ly*i0) for i0 in range(nring)]
		Qloop_list.append( temp )
		print( "Q-loop =", Qloop_list  )
	else:
		Sring_list.append( psi.entanglement_entropy()[int(psi.L/2)-1] )

	
	if 1:
		try:
			temp = ['P1 P2']+['P1 P3']+['Id']*3+['P1 P2']+['P2 P3']+['Id']+['P1 P3']+['P2 P3']
			Pstar = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "Pstar =", Pstar )
			Pstar_list.append( Pstar )
			temp = ['P1 P2']+['P1 P3']+['Id']*3+['P1 P2']+['P2 P3']+['Id']*6+['P1 P2']+['P2 P3']+['Id']+['P1 P3']+['P2 P3']
			P2star = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "P2star =", P2star )
			P2star_list.append( P2star )
			temp = ['P1 P2']+['P1 P3']+['Id']*3+['P1 P2']+['Id']+['P1 P3']+['Id']*2+['P2 P3']+['Id']*2+['P1 P2']+['Id']+['P2 P3']+['P1 P3']+['P2 P3']
			P3star = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "P3star =", P3star )
			P3star_list.append( P3star )

			temp = ['Q3']+['Q2']+['Id']*3+['Q3']+['Q1']+['Id']+['Q2']+['Q1']
			Qhexagon = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "Qhexagon =", Qhexagon )
			Qhexagon_list.append( Qhexagon )
			temp = ['Q3']+['Q2']+['Id']*3+['Q3']+['Q1']+['Id']+['Q1']+['Q3']+['Id']*3+['Q3']+['Q1']+['Id']+['Q2']+['Q1']
			Q2hexagon = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "Q2hexagon =", Q2hexagon )
			Q2hexagon_list.append( Q2hexagon )
			temp = ['Q3']+['Q2']+['Id']*3+['Q3']+['Q2']*2+['Q1']+['Id']+['Q1']+['Id']*2+['Q3']*2+['Q1']+['Q2']+['Q1']
			Q3hexagon = [psi.expectation_value_multi_sites(temp,i0=4+i0*2*Ly)[()] for i0 in range(nring)]
			print( "Q3hexagon =", Q3hexagon )
			Q3hexagon_list.append( Q3hexagon )

			B = Qhexagon[0]
			temp = ['Q3']+['Id']+['Id']*3+['Q3']+['Id']+['Id']+['Q2']
			A = psi.expectation_value_multi_sites(temp,i0=4)[()]
			print( "mini-FM (off-diagonal) =", A/np.sqrt(B) )

			temp = ( ['Id','Q3']+['Id']*(Ly-2)+['Q1']+['Id']*(Ly-1) )*3
			Qstring = [psi.expectation_value_multi_sites(temp,i0=i0*2*Ly)[()] for i0 in range(nring)]
			print( "off-diagonal string =", Qstring ) 
			Qstring_list.append( Qstring )

			temp = ['Id']*(Ly-2)+['P2 P3']+['Id']*(Ly+1)
			if bc=='finite': temp = temp*nring
			else: temp = temp*6
			hor = [psi.expectation_value_multi_sites(temp,i0=i0*2*Ly)[()] for i0 in range(nring)]
			print( "P-string =", hor )
			Pstring_list.append( hor )

			FM_length = 2 #4+6

			FMoffdiag = []
			for i in range(nring):
				temp = ['Q3','Q2']+['Id']*5+['Q3'] + (['Q1']+['Id']*6+['Q3'])*FM_length + ['Q2','Q2']
				string = psi.expectation_value_multi_sites(temp,i0=(2*Ly)*i)[()]
				temp = ['Q3','Q2','Id','Id','Q2','Q2','Id','Q3'] + ['Q1','Id','Q1','Id','Id','Q3','Id','Q3']*FM_length + ['Q2','Q2','Q1','Id','Q2','Q1']
				loop = psi.expectation_value_multi_sites(temp,i0=(2*Ly)*i)[()]
				print(string,loop)
				if np.abs(loop)<1e-11: FMoffdiag.append( 0. )
				else: FMoffdiag.append( string/np.sqrt(np.abs(loop)) )
			print( "Fredenhagen-Marcu order parameter (off-diagonal) =", FMoffdiag ) 
			FMoffdiag_list.append( FMoffdiag )

			FMdiag = []
			for i in range(nring):
				temp = (['P2 P3','P1 P3']+['Id']*6) + (['P2 P3']+['Id']*7)*FM_length + (['P2 P3','Id','Id','Id','P1 P3'])
				string = psi.expectation_value_multi_sites(temp,i0=(2*Ly)*i)[()]
				temp = (['P2 P3','P1 P3','P1 P2','P1 P2']+['Id']*4) + (['P2 P3','Id','Id','P1 P2']+['Id']*4)*FM_length + (['P2 P3','Id','Id','P1 P2','P1 P3','P2 P3'])
				loop = psi.expectation_value_multi_sites(temp,i0=(2*Ly)*i)[()]
				print(string,loop)
				if np.abs(loop)<1e-11: FMdiag.append( 0. )
				else: FMdiag.append( string/np.sqrt(np.abs(loop)) )
			print( "Fredenhagen-Marcu order parameter (diagonal) =", FMdiag )
			FMdiag_list.append( FMdiag )

		except:
			pass
	

	S_list.append( np.max(psi.entanglement_entropy()) )
	n_list.append( np.mean(density(psi)) )
	E_list.append( E0 )

	print( "xi =", xi_list )
	print( "S =", S_list )
	print( "S between rings =", Sring_list )
	print( "mean occupation =", n_list )

	print( "energy density =", E_list )

	print( "\n<n> =", density(psi) )

	Qbottom = []
	Qtop = []
	Pbottom = []
	Ptop = []

	I = []
	S = []
	V = []

	for i in range(nring):
		I += [shift(Pcirc,i*Ly)]
		S += ['P']

	for i in range(nring):
		I += [shift(Qcirc,i*Ly)]
		S += ['Q']

	for i in range(len(Qtop)):
		top = Qtop[i]; bottom = Qbottom[i]
		I += [top,bottom,top+bottom]
		S += ['Q']*3

		print("Q_BFFM =", 0.5*(corr(psi,bottom,'Q') +corr(psi,top,'Q')) / np.sqrt( corr(psi,bottom+top,'Q') ), "for length =", len(top) )

	for i in range(len(Ptop)):
		top = Ptop[i]; bottom = Pbottom[i]
		I += [top,bottom,top+bottom]
		S += ['P']*3

		#print("P_BFFM =", 0.5*(corr(psi,bottom,'P') +corr(psi,top,'P')) / np.sqrt( corr(psi,bottom+top,'P') ), "for length =", len(top) )

	L = Ly

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

		for i in range(1,nring): print("Q_BFFM (i0="+str(i*Ly)+") for "+str(N)+"x"+str(N)+" =", corr(psi,I[-2],S[-2],i0=Ly*i)/np.sqrt(np.abs(corr(psi,I[-1],S[-1],i0=Ly*i))) )

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

		for i in range(1,nring): print("P_BFFM (i0="+str(i*Ly)+") for "+str(N)+"x"+str(N)+" =", corr(psi,I[-2],S[-2],i0=Ly*i)/np.sqrt(np.abs(corr(psi,I[-1],S[-1],i0=Ly*i))) )


	print( "Q_num =", Q_num )
	print( "Q_den =", Q_den )
	print( "P_num =", P_num )
	print( "P_den =", P_den )
	Qnum_list.append( Q_num )
	Qden_list.append( Q_den )
	Pnum_list.append( P_num )
	Pden_list.append( P_den )

	V = []; Snew = []; Inew = []
	for i in range(len(I)):
		try:
			V += [corr(psi,I[i],S[i])]
			Snew += [S[i]]
			Inew += [I[i]]
		except:
			pass
	S = Snew; I = Inew

	if bc=='infinite':
		VBS = [psi.correlation_function('n1','n1',sites1=[0],sites2=[Ly*i])[0,0] - np.mean(density(psi))**2 for i in range(1,10)]
		VBS_list.append(VBS)

		Ploopcorr = [corr(psi,Pcirc+shift(Pcirc,Ly*i),'P') for i in range(10)]
		Ploopcorr_list.append(Ploopcorr)

	res = {
		'chi' : chi_list[:len(E_list)],
		'E' : E_list,
		'Sring' : Sring_list,
		'n' : n_list,
		'n2' : n2_list,
		'xi' : xi_list,
		'P' : Ploop_list,
		'Q' : Qloop_list,
		'Pstring' : Pstring_list,
		'Qstring' : Qstring_list,
		'star' : Pstar_list,
		'2star' : P2star_list,
		'3star' : P3star_list,
		'hexagon' : Qhexagon_list,
		'2hexagon' : Q2hexagon_list,
		'3hexagon' : Q3hexagon_list,
		'FMdiag' : FMdiag_list,
		'FMoffdiag' : FMoffdiag_list,
		'Ploopcorr' : Ploopcorr_list,
		'P_num' : Pnum_list,
		'Q_num' : Qnum_list,
		'P_den' : Pden_list,
		'Q_den' : Qden_list,
		'VBS' : VBS_list,
	}

	if chi>100 and save_res:
		np.save(folder+'res.npy',res)


