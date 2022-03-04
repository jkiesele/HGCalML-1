#!/usr/bin/env python3
import gzip
import pickle
import pandas as pd
import numpy as np
import pyjet
import tqdm

import matplotlib
matplotlib.use("Agg")


g_fontsize=14
def setstyles(fontsize=14):
    global g_fontsize
    g_fontsize=fontsize
    
    axes = {'labelsize': fontsize,
            'titlesize': fontsize}
    
    matplotlib.rc('axes', **axes)
    matplotlib.rc('legend',fontsize=fontsize)
    matplotlib.rc('xtick',labelsize=fontsize)
    matplotlib.rc('ytick',labelsize=fontsize)
    matplotlib.rc('lines', linewidth=2)

setstyles(14)

import uproot3 as uproot
import awkward as ak1

from argparse import ArgumentParser
parser = ArgumentParser('create a jet data frame')
parser.add_argument('peprReco', help='dump of the inference clustering step. Will take predicted and truth particles from here')
parser.add_argument('nanoML', help='source file list used to create the dump of the pepr predictions')
parser.add_argument('outfilePrefix', help='output file prefix')
parser.add_argument('--recreate', help='Recreates data frames', action='store_true')
args = parser.parse_args()


def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.+1e-3)
    
def calc_phi(x, y, z):
    return np.arctan2(y,x)#cms like


def deltaPhi(a,b):
    d = a-b
    d = np.where(d>2.*np.pi, d-2.*np.pi, d)
    return np.where(d<-2.*np.pi,d+2.*np.pi, d)

def pxpy_ptetaphi(pt,phi):
    return pt*np.cos(phi), pt*np.sin(phi)
    
def pxpy_xyze(x,y,z,E):
    eta = calc_eta(x, y, z)
    phi = calc_phi(x, y, z)
    pt = E/np.cosh(eta)
    return pxpy_ptetaphi(pt,phi)
    
#exit()

def get_jets(posX,posY,posZ,E,ptetaphim=False):
    
    if ptetaphim:
        pT=posX
        eta=posY
        phi=posZ
        mass = E
    else:
        eta = calc_eta(posX,posY,posZ)
        phi =calc_phi(posX,posY,posZ)
        pT =  E / np.cosh(eta) #no mass
        mass = pT*0.+0.
    
    p4 = np.zeros(eta.shape[0], dtype={'names':('pT', 'eta', 'phi', 'mass'),
                          'formats':('f8', 'f8', 'f8', 'f8')})
    
    p4['pT']=pT[:,0]
    p4['eta']=eta[:,0]
    p4['phi']=phi[:,0]
    p4['mass']=mass[:,0]
    
    sequence = pyjet.cluster(p4,  algo="antikt", R=0.4)
    return sequence.inclusive_jets()
    
#directly gives matched properties
def get_matches(pred_jets, true_jets, dptrel=.5, DR=.3, add="",minpt=5):
    #pt eta phi 
    p_prop=[]
    for jet in pred_jets:
        if jet.pt < minpt:
            continue
        p_prop.append([jet.pt, jet.eta, jet.phi])
    p_prop = np.array(p_prop)
    
    t_prop=[]
    for jet in true_jets:
        if jet.pt < minpt:
            continue
        t_prop.append([jet.pt, jet.eta, jet.phi])
    t_prop = np.array(t_prop)
    
    matched = []
    
    ep_prop = np.expand_dims(p_prop,axis=1) # P x 1 x 3
    et_prop = np.expand_dims(t_prop,axis=0) # 1 x T x 3
    
    if not ep_prop.shape[0] or not t_prop.shape[0]:
        return pd.DataFrame()
    
    
    DRsq = (ep_prop[...,1]-et_prop[...,1])**2 + deltaPhi(ep_prop[...,2],et_prop[...,2])**2 #P x T
    DpT = np.abs(ep_prop[...,0]-et_prop[...,0])/et_prop[...,0]  #P x T
    
    DRsq = np.where(DpT > dptrel, 100., DRsq)
    DRsq = np.where(DRsq > DR**2, 100., DRsq)
    
    besttruth = np.argmin(DRsq, axis=1)# P truth to pred
    
    assert besttruth.shape[0] == p_prop.shape[0]
    #print('t_prop',t_prop)
    #print('p_prop',p_prop)
    
    dr,t,p =[],[],[]
    for ip,it in enumerate(besttruth):
        if DRsq[ip,it] >  DR**2:
            continue
        dr.append([DRsq[ip,it]])
        t.append([t_prop[it]])
        p.append([p_prop[ip]])
    
    if len(dr)<1:
        return pd.DataFrame()
    dr = np.concatenate(dr,axis=0)
    t = np.concatenate(t,axis=0)
    p = np.concatenate(p,axis=0)

    df = {}
    df[add+'DR'] = np.sqrt(dr)
    df[add+'p_pT'] = (p[:,0])
    df[add+'p_eta'] = (p[:,1])
    df[add+'p_phi'] = (p[:,2])
    df[add+'t_pT'] = (t[:,0])
    df[add+'t_eta'] = (t[:,1])
    df[add+'t_phi'] = (t[:,2])

    df = pd.DataFrame().from_dict(df)
    return df

def create_pepr_data():
    print('creating pepr data')
    
    with gzip.open(args.peprReco, 'rb') as f:
        read_data = pickle.load(f)
    pepr_df = read_data['showers_dataframe']
    
    print(pepr_df.columns)
    Nevents = np.max(pepr_df['event_id'])+1

    njetsp = 0
    njetst = 0
    npart=0
    dfs = []
    Es=[]
    evdfs={
        'p_total_E': [],
        't_total_E': [],
        't_n_jets': [],
        'p_n_jets': [],
        't_n_jets_pt10': [],
        'p_n_jets_pt10': [],
        't_n_jets_pt20': [],
        'p_n_jets_pt20': [],
        't_met_px':[],
        't_met_py':[],
        'p_met_px':[],
        'p_met_py':[],
        't_nav_const': [],
        'p_nav_const': []
        }
    for event in tqdm.tqdm(range(Nevents)):
        e_df = pepr_df[pepr_df['event_id']==event]
        
        pred_df = e_df[e_df['pred_sid'].notnull()]
        true_df = e_df[e_df['truthHitAssignementIdx'].notnull()]
        
        _,c = np.unique(e_df['truthHitAssignementIdx'],return_counts=True)
        #print(c)
        assert np.all(c<2)
        assert np.min(e_df['truthHitAssignementIdx'])>=0
        
        E = np.expand_dims(pred_df['pred_energy'],axis=1)
        posX = np.expand_dims(pred_df['pred_mean_X'],axis=1)
        posY = np.expand_dims(pred_df['pred_mean_Y'],axis=1)
        posZ = np.expand_dims(pred_df['pred_mean_Z'],axis=1)
        
        pred_jets = get_jets(posX,posY,posZ,E)
        njetsp+=len(pred_jets)
        
        
        evdfs['p_nav_const'].append( sum([len(jet) for jet in pred_jets]) / (len(pred_jets)+1e-3) )
        
        evdfs['p_n_jets'].append(len(pred_jets))
        evdfs['p_n_jets_pt20'].append(len([j  for j in pred_jets if j.pt>20 ]))
        evdfs['p_n_jets_pt10'].append(len([j  for j in pred_jets if j.pt>10 ]))
        
        t_E = np.expand_dims(true_df['truthHitAssignedEnergies'],axis=1)#already includes muon correction
        
        evdfs['p_total_E'].append(np.sum(E))
        evdfs['t_total_E'].append(np.sum(t_E))
        
        mpx,mpy = pxpy_xyze(posX, posY, posZ, E)
        evdfs['p_met_px'].append(np.sum(mpx))
        evdfs['p_met_py'].append(np.sum(mpy))
        
        Es.append(t_E)
        
        npart+=t_E.shape[0]
        
        t_posX = np.expand_dims(true_df['truthHitAssignedX'],axis=1)
        t_posY = np.expand_dims(true_df['truthHitAssignedY'],axis=1)
        t_posZ = np.expand_dims(true_df['truthHitAssignedZ'],axis=1)
        
        true_jets = get_jets(t_posX,t_posY,t_posZ,t_E)
        njetst+=len(true_jets)
        
        
        mpx,mpy = pxpy_xyze(t_posX, t_posY, t_posZ, t_E)
        evdfs['t_met_px'].append(np.sum(mpx))
        evdfs['t_met_py'].append(np.sum(mpy))
        
        evdfs['t_n_jets'].append(len(true_jets))
        evdfs['t_n_jets_pt20'].append(len([j  for j in true_jets if j.pt>20 ]))
        evdfs['t_n_jets_pt10'].append(len([j  for j in true_jets if j.pt>10 ]))
        
        evdfs['t_nav_const'].append(sum([len(jet) for jet in true_jets]) / (len(true_jets) + 1e-3))
        
        m = get_matches(pred_jets, true_jets)
        if len(m):
           dfs.append(m)
        
    print('ran on ',Nevents,'endcaps (',Nevents//2,'events)')
    print('true jets',njetst)
    print('pred jets',njetsp)
    print('particles per endcap', npart/Nevents, '(',2.*npart/Nevents,'per event)' )
    pepr_df = pd.concat(dfs)
    
    
    pepr_df.to_pickle(args.outfilePrefix+'_pepr.df.pkl')
    
    Es = np.concatenate(Es,axis=0)[:,0]
    Edf = pd.DataFrame().from_dict({'true_Ep':Es})
    Edf.to_pickle(args.outfilePrefix+'_parts_pepr.df.pkl')
    
    
    #make event quantities event quantities
    for k in  evdfs.keys():
        a = np.array(evdfs[k])
        a = np.reshape(a,[-1,2])
        if "_nav_const" in k:
            evdfs[k] = np.mean(a,axis=-1)
        else:
            evdfs[k] = np.sum(a,axis=-1)
    
    evdfs = pd.DataFrame().from_dict(evdfs)
    evdfs.to_pickle(args.outfilePrefix+'_events_pepr.df.pkl')
    
    return pepr_df, Edf, evdfs

#now use ticl inputs

def readArray(tree, label):#for uproot3/4 ak0 to ak1 transition period
    arr = ak1.from_awkward0(tree[label].array())
    return arr

dfs=[]

def read_file(fname):
    tree = uproot.open(fname)["Events"]
    
    fdfs=[]
    ntpart=0
    aPt = readArray(tree,"PFTICLCand_pt")
    aeta = readArray(tree,"PFTICLCand_eta")
    aphi = readArray(tree,"PFTICLCand_phi")
    amass = readArray(tree,"PFTICLCand_mass")
    aE = aPt * np.cosh(aeta)
    
    tgood = readArray(tree,'MergedSimCluster_isTrainable')
    tE = readArray(tree,"MergedSimCluster_boundaryEnergy")[tgood>0]
    
    tX = readArray(tree,"MergedSimCluster_impactPoint_x")[tgood>0]
    tY = readArray(tree,"MergedSimCluster_impactPoint_y")[tgood>0]
    tZ = readArray(tree,"MergedSimCluster_impactPoint_z")[tgood>0]
    
    atpid = readArray(tree,"MergedSimCluster_pdgId")[tgood>0]
    atbe = readArray(tree,"MergedSimCluster_recEnergy")[tgood>0]
    
    tE = ak1.where(np.abs(atpid)==13,atbe,tE)
    
    Es=[]
    evdfs={
        'p_total_E': [],
        't_total_E': [],
        't_n_jets': [],
        'p_n_jets': [],
        't_n_jets_pt10': [],
        'p_n_jets_pt10': [],
        't_n_jets_pt20': [],
        'p_n_jets_pt20': [],
        't_met_px':[],
        't_met_py':[],
        'p_met_px':[],
        'p_met_py':[],
        't_nav_const': [],
        'p_nav_const': []
        }
    
    njetsp = 0
    njetst = 0
    for e in range(len(aPt)):
        
        ntpart += tE[e].to_numpy().shape[0]
        
        evdfs['p_total_E'].append(np.sum(aE[e].to_numpy()))
        evdfs['t_total_E'].append(np.sum(tE[e].to_numpy()))
        
        mpx,mpy = pxpy_ptetaphi(aPt[e].to_numpy(),aphi[e].to_numpy())
        evdfs['p_met_px'].append(np.sum(mpx))
        evdfs['p_met_py'].append(np.sum(mpy))
        
        
        Es.append(tE[e].to_numpy())
        pred_jets = get_jets(np.expand_dims(aPt[e].to_numpy(),  axis=1),
                             np.expand_dims(aeta[e].to_numpy(), axis=1),
                             np.expand_dims(aphi[e].to_numpy(), axis=1),
                             np.expand_dims(amass[e].to_numpy(),axis=1),ptetaphim=True)
        njetsp+=len(pred_jets)
        
        evdfs['p_n_jets'].append(len(pred_jets))
        evdfs['p_n_jets_pt20'].append(len([j  for j in pred_jets if j.pt>20 ]))
        evdfs['p_n_jets_pt10'].append(len([j  for j in pred_jets if j.pt>10 ]))
        
        evdfs['p_nav_const'].append(sum([len(jet) for jet in pred_jets]) / (len(pred_jets) + 1e-3))
        
        true_jets = get_jets(np.expand_dims(tX[e].to_numpy(),  axis=1),
                             np.expand_dims(tY[e].to_numpy(), axis=1),
                             np.expand_dims(tZ[e].to_numpy(), axis=1),
                             np.expand_dims(tE[e].to_numpy(),axis=1))
        njetst+=len(true_jets)
        m = get_matches(pred_jets, true_jets)
        
        mpx,mpy = pxpy_xyze(
            tX[e].to_numpy(),
            tY[e].to_numpy(),
            tZ[e].to_numpy(),
            tE[e].to_numpy()
            )
        evdfs['t_met_px'].append(np.sum(mpx))
        evdfs['t_met_py'].append(np.sum(mpy))
        
        evdfs['t_n_jets'].append(len(true_jets))
        evdfs['t_n_jets_pt20'].append(len([j  for j in true_jets if j.pt>20 ]))
        evdfs['t_n_jets_pt10'].append(len([j  for j in true_jets if j.pt>10 ]))
        
        evdfs['t_nav_const'].append(sum([len(jet) for jet in true_jets]) / (len(true_jets) + 1e-3))
        
        if len(m):
           fdfs.append(m)
    
    Es = np.concatenate(Es,axis=0)
    
    return fdfs,njetst,njetsp,len(aPt),ntpart,Es,evdfs

def create_ticl_data():
    print('creating ticl data')
    njetst=0
    njetsp=0
    npart=0
    import os
    nanopath = os.path.dirname(args.nanoML)+'/'
    ticldfs=[]
    nevents = 0
    Es=[]
    evdfs={
        'p_total_E': [],
        't_total_E': [],
        't_n_jets': [],
        'p_n_jets': [],
        't_n_jets_pt10': [],
        'p_n_jets_pt10': [],
        't_n_jets_pt20': [],
        'p_n_jets_pt20': [],
        't_met_px':[],
        't_met_py':[],
        'p_met_px':[],
        'p_met_py':[],
        't_nav_const': [],
        'p_nav_const': []
        }
    with open(args.nanoML) as f:
        for fname in tqdm.tqdm(f):
            fname=fname[:-1]
            if len(fname)<1:
                continue
            tdf,nt,npjets,nev,ntpart,pEs,fedfs = read_file(nanopath+fname)
            ticldfs += tdf
            njetst += nt
            njetsp += npjets
            nevents+=nev
            npart+=ntpart
            Es.append(pEs)
            for k in fedfs.keys():
                evdfs[k]+=fedfs[k]
    
    ticl_df = pd.concat(ticldfs)
    print('ticl dataframe from ',nevents,'events (should be npepr/2)')
    print('true jets',njetst)
    print('pred jets',njetsp)
    print('particles per event', npart/nevents)
    ticl_df.to_pickle(args.outfilePrefix+'_ticl.df.pkl')
    
    Es = np.concatenate(Es,axis=0)
    Edf = pd.DataFrame().from_dict({'true_Ep':Es})
    Edf.to_pickle(args.outfilePrefix+'_parts_ticl.df.pkl')
    
    
    evdfs = pd.DataFrame().from_dict(evdfs)
    evdfs.to_pickle(args.outfilePrefix+'_events_ticl.df.pkl')
    
    return ticl_df, Edf,evdfs

#exit()


if args.recreate:
    pepr_df,pepr_Edf,pepr_evdf = create_pepr_data()
    ticl_df,ticl_Edf,ticl_evdf = create_ticl_data()
else:
    pepr_df,pepr_Edf,pepr_evdf  = pd.read_pickle(args.outfilePrefix+'_pepr.df.pkl'), pd.read_pickle(args.outfilePrefix+'_parts_pepr.df.pkl'),pd.read_pickle(args.outfilePrefix+'_events_pepr.df.pkl')
    ticl_df,ticl_Edf,ticl_evdf  = pd.read_pickle(args.outfilePrefix+'_ticl.df.pkl'), pd.read_pickle(args.outfilePrefix+'_parts_ticl.df.pkl'),pd.read_pickle(args.outfilePrefix+'_events_ticl.df.pkl')

print('ticl matches',len(ticl_df))
print('pepr matches',len(pepr_df))

def response(bins, pred, truth, std=False):
    
    pred = np.array(pred)
    truth = np.array(truth)
    x = []
    y=[]
    for i in range(len(bins)-1):
        sel = np.logical_and(truth>=bins[i], truth<bins[i+1])
        
        x.append((bins[i]+bins[i+1])/2.)
        p = pred[sel]
        t = truth[sel]
        r = p/(t+1e-3)
        rm = np.mean(r)
        if std:
            y.append(np.std(r-rm)/rm)
        else:
            y.append(rm)
            
    return x,y


import matplotlib.pyplot as plt

print(len(pepr_df['t_pT']), len(ticl_df['t_pT']))
plt.hist(pepr_df['t_pT'],bins=41,color='tab:orange')
plt.hist(ticl_df['t_pT'],bins=41,label='MSC',alpha=0.5,color='tab:blue')
plt.yscale('log')
plt.legend()
plt.savefig("truth.pdf")
plt.close()

plt.hist(pepr_Edf['true_Ep'],bins=51)
plt.hist(ticl_Edf['true_Ep'],bins=51,label='MSC',alpha=0.5)
plt.yscale('log')
plt.legend()
plt.savefig("truth_part.pdf")
plt.close()


plt.hist(pepr_df['t_pT'],bins=51,label='end-to-end')
plt.hist(ticl_df['t_pT'],bins=51,alpha=0.5)
plt.yscale('log')
plt.xlabel("Jet pT [GeV]")
plt.ylabel("N")
plt.legend()
plt.savefig("Jets_pt.pdf")
plt.close()




#plt.hist(pepr_evdf['t_n_jets'],bins=51,label='truth')
#plt.hist(pepr_evdf['p_n_jets']+0.1,bins=51,label='end-to-end',alpha=0.8)
#plt.hist(ticl_evdf['p_n_jets']-0.1,bins=51,alpha=0.8)
#plt.yscale('log')
#plt.xlabel("Number of Jets per Event")
#plt.ylabel("N")
#plt.legend()
#plt.savefig("N_jets.pdf")
#plt.close()

njetsbins=np.array([-0.5,0.5,1.5,2.5,3.5,4.5])

#plt.hist(ticl_evdf['t_n_jets_pt10'],bins=njetsbins,label='truth',color='tab:green',histtype='step')
#plt.hist(pepr_evdf['p_n_jets_pt10'],bins=njetsbins,label='end-to-end',color='tab:orange',histtype='step')
#plt.hist(ticl_evdf['p_n_jets_pt10'],bins=njetsbins,color='tab:blue',histtype='step')
#plt.yscale('log')
#plt.xlabel("Number of Jets per Event")
#plt.ylabel("N")
#plt.legend()
#plt.savefig("N_jets_pt10.pdf")
#plt.close()
#
#plt.hist(ticl_evdf['t_n_jets_pt20'],bins=njetsbins,label='truth',
#         color='tab:green',histtype='step')
#plt.hist(pepr_evdf['p_n_jets_pt20'],bins=njetsbins,label='end-to-end',color='tab:orange',histtype='step')
#plt.hist(ticl_evdf['p_n_jets_pt20'],bins=njetsbins,color='tab:blue',histtype='step')
#plt.yscale('log')
#plt.xlabel("Number of Jets per Event")
#plt.ylabel("N")
#plt.legend()
#plt.savefig("N_jets_pt20.pdf")
#plt.close()


plt.hist(ticl_evdf['t_nav_const'],bins=51,label='truth',color='tab:green',histtype='step')
plt.hist(pepr_evdf['p_nav_const'],bins=51,label='end-to-end',alpha=1,color='tab:orange',histtype='step')
plt.hist(ticl_evdf['p_nav_const'],bins=51,label='classic', alpha=1.,color='tab:blue',histtype='step')
plt.yscale('log')
plt.xlabel("Average number of constituents per jet in an event")
plt.ylabel("N")
plt.legend()
plt.title(r'CMSSW 12_1_1,     $q\bar{q} \rightarrow t\bar{t}$,     Jets (no $\nu$, $\mu$)')
plt.savefig("N_const.pdf")
plt.close()




plt.hist(pepr_evdf['t_total_E'])
plt.xlabel("True total energy [GeV]")
plt.savefig("event_true_sums.pdf")
plt.close()

print('responses')



sumbins = [300.,500., 600.,700., 800., 900.]
x,y = response(sumbins, pepr_evdf['p_total_E'], pepr_evdf['t_total_E'])
plt.plot(x,y,label='end-to-end',color='tab:orange')
x,y = response(sumbins, ticl_evdf['p_total_E'], ticl_evdf['t_total_E'])
plt.plot(x,y,color='tab:blue')
plt.legend()
plt.xlabel("Total energy [GeV]")
plt.ylabel("reco energy / true energy")
plt.savefig("event_sums.pdf")
plt.close()

x,y = response(sumbins, pepr_evdf['p_total_E'], pepr_evdf['t_total_E'],std=True)
plt.plot(x,y,label='end-to-end',color='tab:orange')
x,y = response(sumbins, ticl_evdf['p_total_E'], ticl_evdf['t_total_E'],std=True)
plt.plot(x,y,color='tab:blue')
plt.legend()
plt.xlabel("Total energy [GeV]")
plt.ylabel(r"$\sigma$(reco energy / true energy)/<reco energy / true energy>")
plt.savefig("event_sums_resolution.pdf")
plt.close()

##MET
def MET(px,py):
    return np.sqrt(px**2+py**2)

metbins = [10., 15., 20.,  30., 50., 100., 200., 400.]

plt.hist(MET(pepr_evdf['t_met_px'], pepr_evdf['t_met_py']))
plt.xlabel("True MET [GeV]")
plt.savefig("met.pdf")
plt.close()


x,y = response(metbins, MET(pepr_evdf['p_met_px'], pepr_evdf['p_met_py']), 
               MET(pepr_evdf['t_met_px'], pepr_evdf['t_met_py']),std=True)
plt.plot(x,y,label='end-to-end',color='tab:orange')
x,y = response(metbins, MET(ticl_evdf['p_met_px'], ticl_evdf['p_met_py']), 
               MET(ticl_evdf['t_met_px'], ticl_evdf['t_met_py']),std=True)
plt.plot(x,y,color='tab:blue')
plt.legend()
plt.xlabel("True forward MET [GeV]")
plt.ylabel(r"$\sigma$(reco MET / true MET)/<reco MET / true MET>")
plt.savefig("met_resolution.pdf")
plt.close()


##
fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict()) # frameon=False removes frames

plt.subplots_adjust(hspace=.0)


ptbins = [5.,  15., 20.,  30., 50., 150., 250.]

#first make sure the truth makes sense
x,y = response(ptbins, pepr_df['p_pT'], pepr_df['t_pT'])
ax1.plot(x,y,label='end-to-end',color='tab:orange')
x,y = response(ptbins, ticl_df['p_pT'], ticl_df['t_pT'])
ax1.plot(x,y,color='tab:blue',label='classic')
#plt.legend()
#ax1.set_xlabel(r"Jet $p_{T}$[GeV]")
ax1.set_ylabel(r"reco $p_T$ / true $p_T$")
#plt.savefig("response.pdf")
#plt.close()


x,y = response(ptbins, pepr_df['p_pT'], pepr_df['t_pT'],std=True)
ax2.plot(x,y,label='end-to-end',color='tab:orange')
x,y = response(ptbins, ticl_df['p_pT'], ticl_df['t_pT'],std=True)
ax2.plot(x,y,color='tab:blue',label='classic')
ax2.set_xlabel("Jet $p_T$ [GeV]")
ax2.set_ylabel(r"$\sigma(p_{T})$ /< $p_{T}$ >")

ax2.set_ylim([0.,0.299])

ax1.grid()
ax2.grid()

ax1.set_title(r'CMSSW 12_1_1,     $q\bar{q} \rightarrow t\bar{t}$,     Jets (no $\nu$, $\mu$)')
plt.legend()
plt.savefig("resolution.pdf")
plt.close()



    
    
