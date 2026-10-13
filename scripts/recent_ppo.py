"""Scheduling-only adaptation of Long et al.'s Lya-HiPPO (TWC 2025).
PPO-Clip chooses grants; prescribed trajectories/no compression fix the other
hierarchy blocks, and minimum feasible transmit power solves the PHY block.
The original age-constraint queue becomes an individual power queue to match
this benchmark. This is explicitly an adaptation, not full semantic reproduction.
Torch is needed only for training; exported NumPy weights support offline inference.
"""
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'results'/'ppo_models'
CONFIG=dict(protocol='scheduling-ppo-v1',train_trace_seeds=list(range(1000,1016)),
    training_seeds=list(range(5)),num_envs=16,rollout_steps=128,updates=500,
    checkpoints=[250,375,500],learning_rate=3e-4,epochs=4,batch_size=256,
    gamma=.99,gae_lambda=.95,clip=.2,entropy=.01,value_coef=.5,max_grad_norm=.5,
    hidden_sizes=[32,32],calibration_seeds=[50,51],calibration_slots=1500,
    full_V=200.,partial_V=6.,torch_threads=1)

def features(age,weight,power,probability,queue,V,cap,pbar,allowed):
    power=np.clip(power,0,100.)
    d=np.minimum(age+1,cap)-1
    return np.stack([age/100.,weight/10.,probability,
        np.log1p(np.clip(power,0,1e6))/np.log(101.),
        queue/V*power/100.,weight*d*probability/100.,
        np.broadcast_to(cap/500.,age.shape),np.broadcast_to(pbar/3.,age.shape)],axis=-1).astype(np.float32)

def _npnet(x,z,prefix):
    for i in range(3):
        x=x@z[f'{prefix}.{2*i}.weight'].T+z[f'{prefix}.{2*i}.bias']
        if i<2:x=np.tanh(x)
    return x

class PPOController:
    def __init__(self,mode,seed=0,checkpoint=None):
        self.mode=mode;self.seed=int(seed)
        meta=json.loads((MODELS/f'{mode}_seed{seed}_selection.json').read_text()) if checkpoint is None else None
        file=MODELS/f'{mode}_seed{seed}_update{meta["selected_update"] if meta else checkpoint}.npz'
        self.file=file; self.weights=dict(np.load(file));self.steps=0
    def decide(self,age,weight,power,probability,queue,V,cap,pbar,allowed,M=3):
        x=features(age,weight,power,probability,queue,V,cap,pbar,allowed)
        logits=_npnet(x,self.weights,'actor').ravel()
        globalx=np.r_[x.mean(0),x.max(0)]
        idle=float(_npnet(globalx,self.weights,'idle')[0])
        ix=np.argsort(-np.where(allowed,logits,-np.inf),kind='stable')[:M]
        return ix[np.asarray(allowed,dtype=bool)[ix] & (logits[ix]>idle)]

class TrainingReplay:
    def __init__(self,mode):
        from run_icc import partial_trace
        import env_and_cmdp as F
        self.mode=mode; E=CONFIG['num_envs'];T=3000;N=12
        self.power=np.zeros((T,E,N));self.weights=np.zeros_like(self.power);self.outcome=np.zeros_like(self.power,dtype=bool)
        self.ln=np.zeros((E,N));self.nl=self.ln.copy();self.pi=self.ln.copy()
        for e,seed in enumerate(CONFIG['train_trace_seeds']):
            if mode=='full':
                env=F.Env(F.clone(F.Cfg(),seed=seed,T=T,V=200))
                for t in range(T):
                    obs=env.observe();env._last=obs
                    self.power[t,e]=obs['pth'];self.weights[t,e]=obs['w'];self.outcome[t,e]=obs['deliverable']
                    env.step([],np.zeros(N))
            else:
                dwell=[2.,8.,20.][e%3]
                lo,dist,pi,ln,nl=partial_trace(seed,T=T,dwell=dwell)
                self.pi[e]=pi;self.ln[e]=ln;self.nl[e]=nl
                rng=np.random.default_rng(91000+seed)
                self.outcome[:,e]=rng.random((T,N))<np.where(lo,.9,.02)
                ev=np.empty((T,N),bool);ev[0]=rng.random(N)<.08
                onset=(1/25)*.08/(1-.08)
                for t in range(1,T):
                    u=rng.random(N);ev[t]=np.where(ev[t-1],u>=1/25,u<onset)
                self.weights[:,e]=np.where(ev,5.,1.)
                fspl=20*np.log10(4*np.pi*2.4e9*dist/3e8)
                sigma=1.38e-23*290*20e6*10**.7*1e3
                self.power[:,e]=3*sigma/10**(-(fspl+1)/10)
        self.V=200. if mode=='full' else 6.;self.cap=500 if mode=='full' else 120
        self.pbar=3. if mode=='full' else .4;self.T=T;self.E=E;self.N=N
        self.reset()
    def reset(self):
        self.t=np.zeros(self.E,dtype=int);self.A=np.ones((self.E,self.N));self.Q=np.zeros_like(self.A);self.th=self.pi.copy()
    def observation(self):
        ids=np.arange(self.E); c=self.power[self.t,ids];w=self.weights[self.t,ids]
        prob=np.ones_like(c) if self.mode=='full' else .9*self.th+.02*(1-self.th)
        allowed=c<=100.
        return features(self.A,w,c,prob,self.Q,self.V,self.cap,self.pbar,allowed),allowed
    def step(self,actions):
        ids=np.arange(self.E); c=self.power[self.t,ids];w=self.weights[self.t,ids]
        selected=np.zeros_like(self.A,dtype=bool)
        for j in range(3):
            a=actions[:,j]; valid=a<self.N;selected[ids[valid],a[valid]]=True
        assert np.all(selected.sum(1)<=3)
        selected &= c<=100.
        p=np.where(selected,c,0.);ack=selected&self.outcome[self.t,ids]
        oldQ=self.Q.copy();self.A=np.where(ack,1.,np.minimum(self.A+1,self.cap))
        # Negative one-slot drift-plus-cost, ignoring action-independent constants.
        reward=-(np.sum(w*self.A+oldQ/self.V*p,axis=1))/100.
        self.Q=np.maximum(self.Q-self.pbar,0.)+p
        if self.mode=='partial':
            ll=np.where(ack,.9,.1);ln=np.where(ack,.02,.98)
            post=np.where(selected,self.th*ll/np.maximum(self.th*ll+(1-self.th)*ln,1e-12),self.th)
            self.th=post*(1-self.ln)+(1-post)*self.nl
        self.t+=1;done=self.t>=self.T
        self.t[done]=0;self.A[done]=1.;self.Q[done]=0.;self.th[done]=self.pi[done]
        return reward.astype(np.float32),done

def train(mode,seed):
    import os
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'results'/'torch_cache'))
    import torch
    from torch import nn
    from torch.distributions import Categorical
    torch.set_num_threads(1);torch.manual_seed(20000+seed);np.random.seed(30000+seed)
    torch.use_deterministic_algorithms(True)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            def mlp(inputs):return nn.Sequential(nn.Linear(inputs,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
            self.actor=mlp(8);self.idle=mlp(16);self.critic=mlp(16)
        def distribution(self,x,allowed,actions=None):
            gx=torch.cat([x.mean(1),x.max(1).values],1)
            logits=torch.cat([self.actor(x).squeeze(-1),self.idle(gx)],1)
            mask=torch.cat([allowed,torch.ones((len(x),1),dtype=torch.bool)],1).clone()
            chosen=[];lp=torch.zeros(len(x));entropy=torch.zeros(len(x))
            ids=torch.arange(len(x))
            for j in range(3):
                dist=Categorical(logits=logits.masked_fill(~mask,-1e9))
                a=dist.sample() if actions is None else actions[:,j]
                lp=lp+dist.log_prob(a);entropy=entropy+dist.entropy();chosen.append(a)
                mask[ids,a]=False;mask[:,-1]=True
            return torch.stack(chosen,1),lp,entropy,self.critic(gx).squeeze(-1)
    env=TrainingReplay(mode);net=Net();optimizer=torch.optim.Adam(net.parameters(),lr=CONFIG['learning_rate'],eps=1e-5)
    MODELS.mkdir(parents=True,exist_ok=True);history=[];tic=time.time()
    S=CONFIG['rollout_steps'];E=CONFIG['num_envs']
    for update in range(1,CONFIG['updates']+1):
        xs=[];ms=[];acts=[];lps=[];vals=[];rews=[];dones=[]
        for step in range(S):
            x,mask=env.observation();xt=torch.from_numpy(x);mt=torch.from_numpy(mask)
            with torch.no_grad():a,lp,ent,v=net.distribution(xt,mt)
            reward,done=env.step(a.numpy())
            xs.append(x);ms.append(mask);acts.append(a.numpy());lps.append(lp.numpy());vals.append(v.numpy());rews.append(reward);dones.append(done)
        x,mask=env.observation()
        with torch.no_grad():_,_,_,nextv=net.distribution(torch.from_numpy(x),torch.from_numpy(mask))
        vals=np.array(vals);rews=np.array(rews);dones=np.array(dones);adv=np.zeros_like(rews);last=np.zeros(E)
        for t in range(S-1,-1,-1):
            nval=nextv.numpy() if t==S-1 else vals[t+1];nonterminal=1.-dones[t]
            delta=rews[t]+CONFIG['gamma']*nval*nonterminal-vals[t]
            last=delta+CONFIG['gamma']*CONFIG['gae_lambda']*nonterminal*last;adv[t]=last
        returns=adv+vals;flatadv=adv.ravel();flatadv=(flatadv-flatadv.mean())/(flatadv.std()+1e-8)
        X=torch.from_numpy(np.array(xs).reshape(S*E,12,8));Mask=torch.from_numpy(np.array(ms).reshape(S*E,12))
        A=torch.from_numpy(np.array(acts).reshape(S*E,3));Old=torch.from_numpy(np.array(lps).ravel())
        Ad=torch.from_numpy(flatadv);Ret=torch.from_numpy(returns.ravel())
        losses=[];kls=[];clips=[]
        for epoch in range(CONFIG['epochs']):
            perm=np.random.permutation(S*E)
            for start in range(0,S*E,CONFIG['batch_size']):
                ids=perm[start:start+CONFIG['batch_size']]
                _,lp,ent,v=net.distribution(X[ids],Mask[ids],A[ids]);ratio=(lp-Old[ids]).exp()
                pg=-torch.minimum(ratio*Ad[ids],torch.clamp(ratio,1-CONFIG['clip'],1+CONFIG['clip'])*Ad[ids]).mean()
                loss=pg+CONFIG['value_coef']*.5*((v-Ret[ids])**2).mean()-CONFIG['entropy']*ent.mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),CONFIG['max_grad_norm']);optimizer.step()
                losses.append(float(loss.detach()));kls.append(float(((ratio-1)-(lp-Old[ids])).mean().detach()));clips.append(float(((ratio-1).abs()>CONFIG['clip']).float().mean()))
        history.append(dict(update=update,samples=update*S*E,reward=float(rews.mean()),loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),clip_fraction=float(np.mean(clips))))
        if update in CONFIG['checkpoints']:
            np.savez_compressed(MODELS/f'{mode}_seed{seed}_update{update}.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
        if update%25==0:
            print(mode,seed,'update',update,'reward',history[-1]['reward'],'seconds',round(time.time()-tic,1),flush=True)
            (MODELS/f'{mode}_seed{seed}_training.json').write_text(json.dumps(dict(config=CONFIG,seed=seed,mode=mode,history=history,elapsed_seconds=time.time()-tic,torch_version=torch.__version__),indent=2))
    print('finished training',mode,seed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['full','partial'],required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();train(args.mode,args.seed)
