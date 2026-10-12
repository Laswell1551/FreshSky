"""UAV channel environment and discretized Tang-CMDP comparison.
Extracted from the existing project; ICC scores and evaluation are in run_icc.py.
"""
import os, time, json, math
import numpy as np
import pandas as pd
from scipy.optimize import linprog

RNG_MASTER = 20270731  # deterministic

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
class Cfg:
    # network
    N = 12                 # UAVs / sources
    M = 3                  # orthogonal uplink sub-bands (M < N)
    T = 3000               # slots per episode
    tau = 0.1              # slot length (s)  -> 100 ms
    # PHY (2.4 GHz WiFi-like) -- ALL POWERS IN mW so value & energy terms are comparable
    fc = 2.4e9
    W = 20e6               # per-sub-band bandwidth (Hz)
    Pmax = 100.0           # max Tx power (mW) = 20 dBm
    pbar_frac = 0.03       # avg-power budget as fraction of Pmax  -> tuned to bind
    Dbits = 4.0e6          # update size (bits)  -> gamma = 2^(D/(W tau)) - 1
    # air-ground channel (Al-Hourani urban)
    H = 100.0              # UAV height (m)
    rmin, rmax = 50.0, 400.0
    a1, a2 = 9.61, 0.16    # Al-Hourani urban LoS-prob constants
    etaL_dB, etaN_dB = 1.0, 20.0   # excess path loss LoS / NLoS (dB)
    mL, mN = 3.0, 1.0      # Nakagami-m LoS / NLoS
    # blockage temporal correlation (mean sojourn in slots)
    mean_sojourn_N = 8.0   # mean NLoS dwell (correlation) -> bursty outages
    # interference: per-link 2-state (low/high) Markov, in mW
    I_low, I_high = 0.0, 8e-9        # mW (high ~ -81 dBm >> noise)
    p_ihigh, mean_sojourn_I = 0.30, 6.0
    # value / event process: w in {1 (normal), wev (event)}, events markovian
    wev = 10.0
    p_event, mean_sojourn_ev = 0.08, 25.0
    Amax = 500
    # Lyapunov knob (power in mW -> V ~ O(1))
    V = 2.0
    # noise
    NF_dB = 7.0
    seed = 0

    def derived(self):
        k, T0 = 1.38e-23, 290.0
        self.sigma2 = k*T0*self.W*(10**(self.NF_dB/10.0))*1e3   # noise power (mW)
        self.gamma = 2**(self.Dbits/(self.W*self.tau)) - 1.0
        self.pbar = self.pbar_frac*self.Pmax
        return self

def clone(cfg, **kw):
    c = Cfg()
    for a in dir(cfg):
        if not a.startswith('_') and not callable(getattr(cfg,a)):
            setattr(c,a,getattr(cfg,a))
    for k,v in kw.items(): setattr(c,k,v)
    return c.derived()

# --------------------------------------------------------------------------
# Channel helpers
# --------------------------------------------------------------------------
def plos(theta_deg, a1, a2):
    return 1.0/(1.0 + a1*np.exp(-a2*(theta_deg - a1)))

def fspl_dB(d, fc):
    c = 3e8
    return 20*np.log10(4*np.pi*fc*d/c)

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Env:
    def __init__(self, cfg):
        self.cfg = cfg.derived() if not hasattr(cfg,'sigma2') else cfg
        self.rng = np.random.default_rng(RNG_MASTER + 1000*cfg.seed)
        c = self.cfg; N = c.N
        # patrol: each UAV oscillates horizontal distance sinusoidally
        self.phase = self.rng.uniform(0, 2*np.pi, N)
        self.peak_radial_speed = self.rng.uniform(6.0, 14.0, N)  # m/s
        amplitude = 0.5*(c.rmax-c.rmin)
        self.freq = self.peak_radial_speed/(2*np.pi*amplitude)  # cycles/second
        # blockage state (0=LoS,1=NLoS), interference (0=low,1=high), event
        self.beta = self.rng.integers(0,2,N)
        self.iis  = (self.rng.random(N) < c.p_ihigh).astype(int)
        self.ev   = (self.rng.random(N) < c.p_event).astype(int)
        self.A = np.ones(N)          # AoI
        self.Q = np.zeros(N)         # virtual energy queue
        self.t = 0
        # for statistical phi(beta): success prob per (uav,beta) estimated online
        self.phi_hits = np.ones((N,2)); self.phi_try = np.ones((N,2))

    def _rh(self, n):
        c=self.cfg
        return c.rmin + 0.5*(c.rmax-c.rmin)*(1+np.sin(2*np.pi*self.freq[n]*(self.t*c.tau) + self.phase[n]))

    def large_scale(self):
        """Return per-UAV large-scale gain L (linear), theta, plos, d."""
        c=self.cfg; N=c.N
        rh = np.array([self._rh(n) for n in range(N)])
        d = np.sqrt(c.H**2 + rh**2)
        theta = np.degrees(np.arctan(c.H/rh))
        eta = np.where(self.beta==0, c.etaL_dB, c.etaN_dB)
        L_dB = fspl_dB(d, c.fc) + eta
        L = 10**(-L_dB/10.0)
        return L, theta, d

    def channel_gain(self, L):
        """Instantaneous gain incl. Nakagami fading (known to full-CSI schedulers)."""
        c=self.cfg; N=c.N
        m = np.where(self.beta==0, c.mL, c.mN)
        # Nakagami power |h|^2 ~ Gamma(shape=m, scale=1/m), mean 1
        fad = self.rng.gamma(shape=m, scale=1.0/m)
        return L*fad

    def interference(self):
        c=self.cfg
        return np.where(self.iis==0, c.I_low, c.I_high)

    def weights(self):
        c=self.cfg
        return np.where(self.ev==0, 1.0, c.wev)

    def p_th(self, g, I):
        c=self.cfg
        return c.gamma*(c.sigma2 + I)/np.maximum(g,1e-30)

    def phi_beta(self):
        """Statistical success prob per UAV given current beta (for Whittle)."""
        return self.phi_hits[np.arange(self.cfg.N), self.beta]/self.phi_try[np.arange(self.cfg.N), self.beta]

    def observe(self):
        L, theta, d = self.large_scale()
        g = self.channel_gain(L)
        I = self.interference()
        pth = self.p_th(g, I)
        deliverable = (pth <= self.cfg.Pmax)
        return dict(L=L, g=g, I=I, pth=pth, deliverable=deliverable,
                    A=self.A.copy(), Q=self.Q.copy(), w=self.weights(),
                    beta=self.beta.copy(), theta=theta)

    def step(self, sched, power):
        """sched: index array of scheduled UAVs (<=M); power: per-UAV Tx power (0 if not sched)."""
        c=self.cfg; N=c.N
        obs = self._last
        s = np.zeros(N, dtype=int)     # delivery indicator
        for n in sched:
            # delivery deterministic given known g & power (full CSI)
            deliver = (power[n]*obs['g'][n]/(c.sigma2+obs['I'][n])) >= c.gamma - 1e-12
            s[n] = 1 if deliver else 0
            # update phi estimate
            self.phi_try[n, self.beta[n]] += 1
            self.phi_hits[n, self.beta[n]] += s[n]
        # AoI update
        self.A = np.where(s==1, 1.0, np.minimum(self.A+1, c.Amax))
        # energy virtual queue
        p = np.array([power[n] for n in range(N)])
        self.Q = np.maximum(self.Q - c.pbar, 0.0) + p
        # exogenous dynamics: blockage / interference / event Markov transitions
        self._advance_markov()
        self.t += 1
        return s, p

    def _advance_markov(self):
        c=self.cfg; N=c.N
        # blockage: stationary LoS prob from Al-Hourani at current theta
        _, theta, _ = self.large_scale()
        piL = plos(theta, c.a1, c.a2)                      # target stationary LoS prob
        pNL = 1.0/c.mean_sojourn_N                         # NLoS->LoS
        pLN = np.clip(pNL*(1-piL)/np.maximum(piL,1e-3), 0, 1)  # LoS->NLoS
        u = self.rng.random(N)
        newbeta = self.beta.copy()
        los = (self.beta==0); nlos=~los
        newbeta[los]  = np.where(u[los]  < pLN[los], 1, 0)
        newbeta[nlos] = np.where(u[nlos] < pNL,      0, 1)
        self.beta = newbeta
        # interference 2-state
        pIH = c.p_ihigh; pI_HL = 1.0/c.mean_sojourn_I
        pI_LH = np.clip(pI_HL*pIH/max(1-pIH,1e-3),0,1)
        u=self.rng.random(N); low=(self.iis==0)
        self.iis = np.where(low, (u<pI_LH).astype(int), (u>=pI_HL).astype(int))
        # event process
        pE=c.p_event; pE_off=1.0/c.mean_sojourn_ev
        pE_on=np.clip(pE_off*pE/max(1-pE,1e-3),0,1)
        u=self.rng.random(N); noe=(self.ev==0)
        self.ev = np.where(noe, (u<pE_on).astype(int), (u>=pE_off).astype(int))

class Policy:
    name="base"; ref=""
    def __init__(self,cfg): self.cfg=cfg
    def decide(self, env, obs): raise NotImplementedError

class TangCMDP(Policy):
    """Truncated constrained-Markov-decision-process policy of Tang et al.
    (IEEE JSAC 2020), adapted to the present air--ground channel.

    Required power is quantized into four deliverable channel states plus one
    blocked state.  A finite-state occupancy-measure linear program is solved
    offline for every UAV.  A common bandwidth multiplier is found by bisection,
    and the two bracketing stationary policies are mixed exactly as in the
    relaxed construction.  At run time, requesting UAVs are truncated uniformly
    at random to the M available sub-bands.  The baseline uses plain AoI, as the
    published method does, and never observes future samples or the test trace.
    """
    name="Tang-CMDP [JSAC'20]"
    ref="Tang et al., IEEE JSAC 2020 (offline CMDP/LP with random truncation)"
    XMAX=40
    QBINS=4
    TRAIN_SLOTS=6000
    _CACHE={}

    def __init__(self,cfg):
        super().__init__(cfg)
        self.rng=np.random.default_rng(RNG_MASTER+23000+cfg.seed)
        key=(cfg.seed,cfg.N,cfg.M,cfg.pbar,cfg.Pmax,cfg.H,cfg.rmin,cfg.rmax,
             cfg.mean_sojourn_N,cfg.p_ihigh,cfg.mean_sojourn_I,cfg.I_high,
             cfg.Dbits,cfg.W,cfg.tau)
        if key not in self._CACHE:
            self._CACHE[key]=self._build_offline_policy(cfg)
        self.edges,self.xi,self.offline_price,self.offline_activation=self._CACHE[key]

    @classmethod
    def _fit_channel_model(cls,cfg):
        # Mission geometry (phase and patrol rate) is known offline, as assumed
        # by the statistical CMDP baseline.  Fading and Markov-state samples
        # use an independent stream, so the evaluated trace is never replayed.
        train=clone(cfg,T=cls.TRAIN_SLOTS)
        env=Env(train)
        env.rng=np.random.default_rng(RNG_MASTER+731000+cfg.seed)
        env.beta=env.rng.integers(0,2,train.N)
        env.iis=(env.rng.random(train.N)<train.p_ihigh).astype(int)
        env.ev=(env.rng.random(train.N)<train.p_event).astype(int)
        trace=[]
        zero={n:0.0 for n in range(train.N)}
        for _ in range(cls.TRAIN_SLOTS):
            obs=env.observe(); env._last=obs; trace.append(obs['pth'].copy())
            env.step([],zero)
        trace=np.asarray(trace)
        edges=np.zeros((train.N,cls.QBINS-1))
        labels=np.zeros_like(trace,dtype=int)
        omega=np.zeros((train.N,cls.QBINS+1))
        trans=np.zeros((train.N,cls.QBINS+1,cls.QBINS+1))
        for n in range(train.N):
            good=trace[:,n]<=train.Pmax
            vals=trace[good,n]
            if len(vals)<cls.QBINS:
                edges[n]=np.linspace(0.25,0.75,cls.QBINS-1)*train.Pmax
            else:
                edges[n]=np.quantile(vals,np.arange(1,cls.QBINS)/cls.QBINS)
            q=np.where(good,np.digitize(trace[:,n],edges[n]),cls.QBINS)
            labels[:,n]=q
            for j in range(cls.QBINS):
                v=trace[q==j,n]
                omega[n,j]=float(np.mean(v)) if len(v) else train.Pmax
            omega[n,cls.QBINS]=0.0
            counts=np.full((cls.QBINS+1,cls.QBINS+1),1e-3)
            np.add.at(counts,(q[:-1],q[1:]),1.0)
            trans[n]=counts/counts.sum(axis=1,keepdims=True)
        return edges,omega,trans

    @classmethod
    def _solve_one(cls,P,omega,pbar,price):
        X,Q=cls.XMAX,cls.QBINS+1; S=X*Q
        # z=[mu(0:S), y(0:S)], where y is active occupancy.
        cost=np.zeros(2*S)
        for x in range(X): cost[x*Q:(x+1)*Q]=x+1
        cost[S:]=price
        Aeq=[]; beq=[]
        for xp in range(X):
            for qp in range(Q):
                row=np.zeros(2*S); row[xp*Q+qp]=1.0
                for x in range(X):
                    xn=min(x+1,X-1)
                    for q in range(Q):
                        k=x*Q+q; prob=P[q,qp]
                        if xn==xp: row[k]-=prob
                        if xp==0: row[S+k]-=prob
                        if xn==xp: row[S+k]+=prob
                Aeq.append(row); beq.append(0.0)
        row=np.zeros(2*S); row[:S]=1.0
        Aeq.append(row); beq.append(1.0)
        Aub=[]; bub=[]
        # Active occupancy cannot exceed total occupancy.
        for k in range(S):
            row=np.zeros(2*S); row[S+k]=1.0; row[k]=-1.0
            Aub.append(row); bub.append(0.0)
        row=np.zeros(2*S)
        for x in range(X): row[S+x*Q:S+(x+1)*Q]=omega
        Aub.append(row); bub.append(pbar)
        bounds=[(0.0,1.0)]*(2*S)
        for x in range(X): bounds[S+x*Q+cls.QBINS]=(0.0,0.0)
        sol=linprog(cost,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),
                    A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,
                    method='highs')
        if not sol.success:
            raise RuntimeError("Tang-CMDP offline LP failed: "+sol.message)
        mu=sol.x[:S].reshape(X,Q); y=sol.x[S:].reshape(X,Q)
        return mu,y,float(y.sum())

    @classmethod
    def _build_offline_policy(cls,cfg):
        edges,omega,trans=cls._fit_channel_model(cfg)
        def solve_all(price):
            out=[cls._solve_one(trans[n],omega[n],cfg.pbar,price) for n in range(cfg.N)]
            return out,sum(v[2] for v in out)
        low,lo_act=solve_all(0.0)
        if lo_act<=cfg.M+1e-8:
            mixed=low; price=0.0; activation=lo_act
        else:
            wl,wh=0.0,1.0; high,hi_act=solve_all(wh)
            while hi_act>cfg.M and wh<1e6:
                wl=wh; low,lo_act=high,hi_act; wh*=2.0
                high,hi_act=solve_all(wh)
            for _ in range(9):
                wm=0.5*(wl+wh); mid,mid_act=solve_all(wm)
                if mid_act>cfg.M: wl,low,lo_act=wm,mid,mid_act
                else: wh,high,hi_act=wm,mid,mid_act
            nu=np.clip((cfg.M-hi_act)/(lo_act-hi_act+1e-12),0.0,1.0)
            mixed=[]
            for a,b in zip(low,high):
                mu=nu*a[0]+(1-nu)*b[0]; y=nu*a[1]+(1-nu)*b[1]
                mixed.append((mu,y,float(y.sum())))
            price=nu*wl+(1-nu)*wh; activation=sum(v[2] for v in mixed)
        xi=[]
        for mu,y,_ in mixed:
            ratio=np.divide(y,mu,out=np.zeros_like(y),where=mu>1e-8)
            ratio=np.clip(ratio,0.0,1.0)
            # Tang et al., Eq. (32): states outside the positive recurrent
            # support inherit the threshold action instead of becoming an
            # accidental never-transmit state after numerical LP truncation.
            for q in range(cls.QBINS):
                for x in range(cls.XMAX):
                    if x==cls.XMAX-1 or mu[x,q]<=1e-8 or (x>0 and ratio[x-1,q]>=1-1e-8):
                        ratio[x,q]=1.0
            ratio[:,cls.QBINS]=0.0
            xi.append(ratio)
        return edges,np.asarray(xi),float(price),float(activation)

    def decide(self,env,obs):
        q=np.empty(self.cfg.N,dtype=int)
        for n in range(self.cfg.N):
            q[n]=self.QBINS if not obs['deliverable'][n] else np.digitize(obs['pth'][n],self.edges[n])
        x=np.minimum(obs['A'].astype(int),self.XMAX)-1
        request=self.rng.random(self.cfg.N)<self.xi[np.arange(self.cfg.N),x,q]
        cand=np.flatnonzero(request & obs['deliverable'])
        if len(cand)>self.cfg.M:
            cand=self.rng.choice(cand,size=self.cfg.M,replace=False)
        power={n:0.0 for n in range(self.cfg.N)}
        for n in cand: power[int(n)]=float(obs['pth'][n])
        return [int(n) for n in cand],power

