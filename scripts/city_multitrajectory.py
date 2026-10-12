"""Five independently generated patrol sets per cached OSM city.
Each geometry is calibrated on its own prefix and evaluated with five paired delivery/event seeds.
Paths stay in the service area and avoid height-grid obstacles with 5m clearance.
Vector ray marching uses exactly the original grid and link-dependent sample counts.
"""
from run_icc import *
import geometry_builder as G
GEO_SEEDS=range(1,6); DELIVERY_SEEDS=range(5)
def patrol_positions(H,cell,sx,sy,seed,N=12,T=2000):
    from scipy.ndimage import label
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    rng=np.random.default_rng(seed); pos=np.zeros((T,N,3)); speeds=[]; clearances=[]
    inside=((np.arange(H.shape[0])+.5)*cell<sx)[:,None]&((np.arange(H.shape[1])+.5)*cell<sy)[None,:]
    for n in range(N):
        altitude=rng.uniform(80,120); free=(H+5<altitude)&inside; labels,count=label(free)
        sizes=np.bincount(labels.ravel()); sizes[0]=0; component=labels==int(np.argmax(sizes)); nodes=np.flatnonzero(component)
        ix,iy=np.unravel_index(nodes,H.shape); coords=np.column_stack([ix,iy]); index=np.full(H.shape,-1,int); index.ravel()[nodes]=np.arange(len(nodes))
        src=[]; dst=[]
        for a,b in [(index[:-1,:],index[1:,:]),(index[:,:-1],index[:,1:])]:
            ok=(a>=0)&(b>=0); src.append(a[ok]);dst.append(b[ok])
        rr=np.concatenate(src);cc=np.concatenate(dst); graph=coo_matrix((np.ones(2*len(rr)),(np.r_[rr,cc],np.r_[cc,rr])),shape=(len(nodes),len(nodes))).tocsr()
        current=int(rng.integers(len(nodes))); speed=rng.uniform(6,14); speeds.append(float(speed)); route=[current]
        need=speed*.1*(T-1)
        while (len(route)-1)*cell<need+cell:
            distances,previous=dijkstra(graph,directed=False,indices=current,return_predecessors=True)
            candidates=np.flatnonzero(np.isfinite(distances)&(distances>=100/cell))
            assert len(candidates)>0
            target=int(rng.choice(candidates)); segment=[target]
            while segment[-1]!=current:
                parent=int(previous[segment[-1]]); assert parent>=0; segment.append(parent)
            segment.reverse(); route.extend(segment[1:]); current=target
        route=np.array(route); walked=np.arange(T)*speed*.1/cell; part=np.floor(walked).astype(int); fraction=walked-part
        centers=(coords[route]+.5)*cell
        pos[:,n,:2]=centers[part]+fraction[:,None]*(centers[part+1]-centers[part]); pos[:,n,2]=altitude
        gi=(pos[:,n,0]//cell).astype(int); gj=(pos[:,n,1]//cell).astype(int)
        clearance=pos[:,n,2]-H[gi,gj]; assert clearance.min()>=5
        assert np.max(np.linalg.norm(np.diff(pos[:,n,:],axis=0),axis=1))<=speed*.1+1e-8
        clearances.append(float(clearance.min()))
    return pos,speeds,clearances

def traces(H,cell,gs,pos):
    T,N,_=pos.shape; los=np.zeros((T,N),bool); dist=np.zeros((T,N))
    for n in range(N):
        p=pos[:,n,:]; seg=np.linalg.norm(p-gs,axis=1); steps=np.clip(seg/(cell*.7),80,400).astype(int); blocked=np.zeros(T,bool)
        for k in np.unique(steps):
            ids=np.flatnonzero(steps==k); alpha=np.linspace(.02,.98,k); ray=p[ids,None,:]+alpha[None,:,None]*(gs-p[ids,None,:])
            ix=(ray[:,:,0]//cell).astype(int); iy=(ray[:,:,1]//cell).astype(int); valid=(ix>=0)&(ix<H.shape[0])&(iy>=0)&(iy<H.shape[1])
            heights=H[np.clip(ix,0,H.shape[0]-1),np.clip(iy,0,H.shape[1]-1)]
            blocked[ids]=np.any(valid&(heights>ray[:,:,2]),axis=1)
        los[:,n]=~blocked; dist[:,n]=seg
    return los,dist
def main():
    tic=time.time(); rows=[]; meta=[]
    for city in ['midtown_manhattan','city_of_london']:
        bounds=(40.7485,-73.9911,40.7593,-73.9769) if city=='midtown_manhattan' else (51.5100,-.0900,51.5200,-.0740)
        G.MINLAT,G.MINLON,G.MAXLAT,G.MAXLON=bounds
        raw=json.loads((ROOT/'data'/('osm_'+city+'.json')).read_text(encoding='utf8')); polys,heights,src=G.parse_buildings(raw); H,cell,sx,sy,_=G.build_grid(polys,heights); gs,_=G.place_gs(H,cell)
        for geo in GEO_SEEDS:
            file=ROOT/'data'/f'osm_patrol_{city}_geo{geo}.npz'
            if file.exists():
                z=np.load(file); lo,di,pos=z['los'],z['dist'],z['positions']; speeds=z['speeds'].tolist(); clearances=z['clearances'].tolist()
            else:
                pos,speeds,clearances=patrol_positions(H,cell,sx,sy,geo); lo,di=traces(H,cell,gs,pos)
                np.savez_compressed(file,los=lo,dist=di,positions=pos,speeds=speeds,clearances=clearances)
            for n in range(12):
                ids=np.arange(0,2000,137); exact=np.array([not G.los_blocked(*pos[t,n],*gs,H,cell,sx,steps=int(np.clip(di[t,n]/(cell*.7),80,400))) for t in ids]); assert np.array_equal(exact,lo[ids,n])
            train=lo[:300]; pi=np.clip(train.mean(0),.05,.95); nL=train[:-1].sum(0); nN=(~train[:-1]).sum(0)
            ln=((train[:-1]&~train[1:]).sum(0)+1)/(nL+2); nl=((~train[:-1]&train[1:]).sum(0)+1)/(nN+2)
            for seed in DELIVERY_SEEDS:
                for pol in ['FreshSky-B','Memoryless']:
                    r=partial_episode(pol,seed,lo,di,pi,ln,nl,pbar=1.6,label=city,radio_reserve_mJ=.1*1.6*1700); r.update(geometry_seed=geo); rows.append(r)
            meta.append(dict(city=city,geometry_seed=geo,LoS=float(lo.mean()),min_dist=float(di.min()),max_dist=float(di.max()),min_clearance_m=min(clearances),speeds_mps=speeds,gs_xyz_m=gs.tolist(),trace_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            print('done city',city,'geometry',geo,flush=True)
    df=pd.DataFrame(rows); df.to_csv(RES/'city_multitrajectory_seeds.csv',index=False)
    cols=['wAoI','raw_wAoI','saturation','max_power','mean_power','psi_mJ','battery_min_mJ','guard_rejections']
    trajectory=df.groupby(['setting','policy','geometry_seed'],sort=False)[cols].mean().reset_index(); trajectory.to_csv(RES/'city_multitrajectory_means.csv',index=False)
    q=trajectory.groupby(['setting','policy'],sort=False)[cols].agg(['mean','std']); q.columns=['_'.join(x) for x in q.columns]; q.reset_index().to_csv(RES/'city_multitrajectory_summary.csv',index=False)
    manifest=dict(geometry_seeds=list(GEO_SEEDS),delivery_event_seeds=list(DELIVERY_SEEDS),replicates_per_policy_city=25,uncertainty_unit='SD across five independent geometry means, each averaging five event/delivery seeds',calibration_slots=300,evaluation_slots=1700,radio_reserve_mJ=.1*1.6*1700,remaining_energy_guard=True,grid_cell_m=6,trajectory_meta=meta,trajectory_model='independent collision-avoiding free-grid routes, 5m height-grid clearance, altitude 80-120m, speed 6-14m/s, no boundary wrapping',ray_march='vectorized original link-dependent sample counts; scalar checkpoints agree',runtime_seconds=time.time()-tic)
    (RES/'city_multitrajectory_manifest.json').write_text(json.dumps(manifest,indent=2)); print(q.reset_index().to_string(index=False),flush=True)
if __name__=='__main__': main()
