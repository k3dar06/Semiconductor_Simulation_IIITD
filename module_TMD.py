#Copyright (c) 2004-2015, G. Fiori, G. Iannaccone, University of Pisa.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    - Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#    - Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#    - All advertising materials mentioning features or use of this software
#        must display the following acknowledgement:
#        This product includes software developed by G.Fiori and G.Iannaccone at
#        University of Pisa.
#    - Neither the name of the University of Pisa nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE AUTHORS
#BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#                       SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#                       INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#THE POSSIBILITY OF SUCH DAMAGE.


from NanoTCAD_ViDES import *
class TMD:
    
    def __init__(self,L,n_or_p):
        #degeneracy of 2 is already taken into account
        #in the Hamiltonian, since the minimum is in Kf,
        #so by itself degeneracy is 2 as in graphene
        #if you specify deg=2, then total degeneracy is 4
        # Effective masses [units of m0]. Stored ON THE OBJECT (self.me/self.mh),
        # not as local variables, so consumers outside this file can read the
        # material's actual masses instead of duplicating the literals. E_band.py's
        # intrinsic-level formula needs the mh/me ratio; when these were locals it
        # had 0.659/0.572 hardcoded, which silently would NOT track a change of
        # material here.
        self.me=0.572;
        self.mh=0.659
        self.Egap=1.78
        self.deg=1;
        self.n=1;
        ymin=-20;
        ymax=L+20;
        self.acc=0.2;
        self.Nc=int(4*(floor((floor(L/self.acc)-1)/3)));
        self.Phi=zeros(self.Nc);
        self.Ei=zeros(self.Nc);
        self.Eupper=1000.0;
        self.Elower=-1000.0;
        self.delta=sqrt(3)*self.acc;
        self.kmax=pi/self.delta;
        self.kmin=0;
        self.dk=0.1;
        self.dE=1e-3;
        self.thop=-2.59;
        if (n_or_p=="n"):
            self.thopBN=-sqrt(2*q*self.Egap/(3*(self.acc*1e-9*sqrt(3))**2*m0*self.me))*hbar/q;
        else:
            self.thopBN=-sqrt(2*q*self.Egap/(3*(self.acc*1e-9*sqrt(3))**2*m0*self.mh))*hbar/q;
        self.eta=1e-5;
        self.mu1=0.0;
        self.mu2=0.0;
        self.Temp=300;
        self.E=zeros(int(NEmax));
        self.T=zeros(int(NEmax));
        self.charge=zeros(self.Nc);
        self.rank=0;
        self.atoms_coordinates();
        self.gap();
        self.T2D="no"
        # ===== ADDED (LDOS patch): opt-in switch for the local density of states.
        # Default "no" -> zero cost for scripts that never read self.ldos (e.g. the
        # long I-V sweeps in 2d_NanoScale_FETS.py). Computing it means building and
        # k-accumulating a (Nc, NEmax) array (~4.5 MB) for EVERY k-point of EVERY
        # self-consistent iteration, which is pure waste if nobody reads it.
        # Set MoS2.compute_ldos="yes" (as E_band.py does) to get self.ldos/self.ne
        # and hence the genuine NEGF quasi-Fermi level.
        self.compute_ldos="no"
        # ===== END ADDED =====
        self.ymin=ymin;
        self.ymax=ymax;
        self.BC_MX2=0.2;
        self.BV_MX2=0.2-self.Egap;
        self.atoms_coordinates();
    def atoms_coordinates(self):
        GNR_atoms_coordinates(self);
        self.y=array(self.z);
        self.x=zeros(size(self.y));
        return;
    def gap(self):
        return self.Egap;
    def charge_T(self):
        # Number of slices and atoms
        slices=self.Nc;
        atoms=1;
        # I define the vector of the k-wave vector
        kvect=arange(self.kmin,self.kmax,self.dk)
        # I start defining the Hamiltonian for the graphene flake
        h=zeros((2*slices,3),dtype=complex);
        h[0][0]=1;
        for i in range(1,slices+1):
            h[i][0]=i
            h[i][1]=i
            if ((self.y[i-1]>=self.ymin)&(self.y[i-1]<=self.ymax)):
                if ((i%2)==0):
                    h[i][2]=self.BC_MX2;
                else:
                    h[i][2]=self.BV_MX2;

        kk=1;
        for ii in range(slices+1,2*slices):
            if ((ii%2)==1):
                h[ii][0]=kk;
                h[ii][1]=kk+1;
                if ((self.y[kk-1]>=self.ymin)&(self.y[kk-1]<=self.ymax)):
                    h[ii][2]=self.thopBN;
                else:
                    h[ii][2]=self.thop; 
            kk=kk+1;
        
        # I then compute the charge and the T for each energy and k and perform the integral
        i=0;
        k=self.kmin;
        H = Hamiltonian(atoms, slices)
        # ===== ADDED (LDOS patch): apply the compute_ldos switch =====
        # The C side (H_charge_T) tests PyObject_HasAttrString(obj,"ldos") to decide
        # whether to compute and return the LDOS. So removing these attributes is the
        # switch: the C then skips that work entirely and no (Nc, NEmax) array is
        # built per k-point. Hamiltonian.charge_T guards its re-wrap with hasattr,
        # so it stays happy when they are gone.
        ldos_on = (getattr(self,"compute_ldos","no")=="yes")
        # Allocate the LDOS return arrays ONLY when they are actually wanted.
        # Hamiltonian.__init__ no longer pre-allocates them (that churned a large
        # array every SC iteration and fragmented the heap). Creating them here,
        # once, right before the k-loop, means:
        #   ldos_on == False  -> H has no ldos/ne/NE, the C skips all LDOS work
        #                        (it gates on HasAttrString), and nothing large is
        #                        allocated -- the fix for the sweep's memory growth.
        #   ldos_on == True   -> exactly the (n*Nc, NEmax) / (n*Nc,) arrays the C
        #                        needs, same as before, just created explicitly.
        # H was built as Hamiltonian(atoms, slices), so its site count is
        # atoms*slices (atoms==1 here). NEmax picks up any override set before the
        # run (e.g. raised at low temperature).
        if (ldos_on):
            H.ldos = zeros((atoms*slices, int(NEmax)))
            H.ne   = zeros(atoms*slices)
            H.NE   = 0
        # ===== END ADDED =====
        if (self.T2D=="yes"):
            EE=arange(self.Elower,self.Eupper,self.dE);
            kvect=arange(self.kmin,self.kmax+self.dk,self.dk);
            X,Y=meshgrid(EE,kvect);
            Z=zeros((size(EE),size(kvect)))
        while (k<=(self.kmax+self.dk*0.5)):
            if (self.rank==0): print("----------------------------------")
            if (self.rank==0): print(("    kx range: [%s,%s] ") %(self.kmin,self.kmax));
            if (self.rank==0): print(("    iteration %s ") %i);
            if (self.rank==0): print("----------------------------------")
            flaggo=0;
            kk=1;
            # I fill the Hamiltonian for the actual wavevector k in the cycle
            for ii in range(slices+1,2*slices):
                if ((ii%2)==0):
                    h[ii][0]=kk;
                    h[ii][1]=kk+1;
                    if ((flaggo%2)==0):
                        if ((self.y[kk-1]>=self.ymin)&(self.y[kk-1]<=self.ymax)):
                            h[ii][2]=self.thopBN+self.thopBN*exp(k*self.delta*1j);
                        else:
                            h[ii][2]=self.thop+self.thop*exp(k*self.delta*1j);
                    else:
                        if ((self.y[kk-1]>=self.ymin)&(self.y[kk-1]<=self.ymax)):
                            h[ii][2]=self.thopBN+self.thopBN*exp(-k*self.delta*1j);
                        else:
                            h[ii][2]=self.thop+self.thop*exp(-k*self.delta*1j);
                    flaggo=flaggo+1;
                kk=kk+1;
                
            H.Eupper = self.Eupper;
            H.Elower = self.Elower;
            H.rank=self.rank;
            H.H = h
            H.dE=self.dE;
            H.Phi=self.Phi;
            H.Ei=-self.Phi;
            H.eta=self.eta;
            H.mu1=self.mu1;
            H.mu2=self.mu2;
            H.Egap=self.gap();
            # ===== ADDED (temperature fix) =====
            # Temp was NOT being forwarded. The C core reads it off THIS object
            # (H_charge_T does PyObject_GetAttrString(obj,"Temp") and builds
            # vt = kB*Temp/q from it), while Hamiltonian.__init__ hardcodes
            # self.Temp = 300. So every Fermi occupation inside the NEGF charge
            # integral ran at 300 K no matter what was set on the TMD object,
            # even though TMD.current() and the Poisson side DID honour self.Temp
            # -- the simulation was silently running at two different
            # temperatures at once. Forwarding it makes the whole solve consistent.
            H.Temp=self.Temp;
            # ===== END ADDED =====
            
            
            # I then compute T and the charge for the actual kx
            H.charge_T()
            
            # I sum up all the contribution
            if (i==0):
                self.E=H.E;
                # the factor 2 is because I integrate over kx>0
                self.T=self.deg*H.T*(2*self.dk/(2*pi));
                self.charge=self.deg*H.charge*(2*self.dk/(2*pi));
                # ===== ADDED (LDOS patch): accumulate the local DOS over k with
                # the SAME weight as the charge, and record the valid energy count
                # NE (same for every k since Eupper/Elower depend only on Phi). =====
                if (ldos_on):
                    self.ldos=self.deg*H.ldos*(2*self.dk/(2*pi));
                    self.ne=self.deg*H.ne*(2*self.dk/(2*pi));
                    self.NE=H.NE;
                # ===== END ADDED =====
            else:
                # the factor 2 is because I integrate over kx>0
                self.T=self.T+self.deg*H.T*(2*self.dk/(2*pi));
                self.charge=self.charge+self.deg*H.charge*(2*self.dk/(2*pi));
                # ===== ADDED (LDOS patch) =====
                if (ldos_on):
                    self.ldos=self.ldos+self.deg*H.ldos*(2*self.dk/(2*pi));
                    self.ne=self.ne+self.deg*H.ne*(2*self.dk/(2*pi));
                # ===== END ADDED =====

            if (self.T2D=="yes"):
                print(size(Z[:,i]),size(H.T),size(EE));
                Z[:,i]=H.T[:size(EE)];
            k=k+self.dk
            i=i+1;

        if (self.T2D=="yes"):
            plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,
                       origin='lower', extent=[self.kmin,self.kmax,self.Elower,self.Eupper])
            show()

        del H;
        self.E=array(self.E);
        self.T=array(self.T)*1e9;
        #        self.charge=array(self.charge)*1e9;
        temporary=zeros(size(self.charge));
        #Let's give back the average charge within 2 near atoms
        for ii in range(0,self.Nc,2):
            temporary[ii]=(self.charge[ii]+self.charge[ii+1])*0.5*1e9;
            temporary[ii+1]=(self.charge[ii]+self.charge[ii+1])*0.5*1e9;
        self.charge=temporary
        # ===== ADDED (LDOS patch): apply the SAME 1e9 scaling and 2-atom
        # pair-averaging to ldos as to charge, so the identity
        #     n(site) = sum_E ldos(site,E) * Fermi((E-Ef)/vt) * dE
        # holds site-by-site. This is what lets E_band.py recover the genuine
        # NEGF electron quasi-Fermi level by inverting that relation per site. =====
        if (ldos_on):
            self.ldos=array(self.ldos)
            ldos_temp=zeros(self.ldos.shape)
            for ii in range(0,self.Nc,2):
                ldos_temp[ii]  =(self.ldos[ii]+self.ldos[ii+1])*0.5*1e9
                ldos_temp[ii+1]=(self.ldos[ii]+self.ldos[ii+1])*0.5*1e9
            self.ldos=ldos_temp
        # self.ne (true electron density) gets the identical treatment, so that
        #     ne(site) = sum_E ldos(site,E)*Fermi((E-Ef)/vt)*dE
        # stays exact site-by-site -- including in the depleted channel, where
        # |self.charge| (net, holes-electrons) would have been wrong.
            self.ne=array(self.ne)
            ne_temp=zeros(size(self.ne))
            for ii in range(0,self.Nc,2):
                ne_temp[ii]  =(self.ne[ii]+self.ne[ii+1])*0.5*1e9
                ne_temp[ii+1]=(self.ne[ii]+self.ne[ii+1])*0.5*1e9
            self.ne=ne_temp
        # ===== END ADDED =====
        #        savetxt("ncar.temp",self.charge)
        #    exit(0);
        del kvect,h;
        return;

    def current(self):
        vt=kboltz*self.Temp/q;
        E=array(self.E);
        T=array(self.T);
        arg=2*q*q/(2*pi*hbar)*T*(Fermi((E-self.mu1)/vt)-Fermi((E-self.mu2)/vt))*self.dE
        return sum(arg);


