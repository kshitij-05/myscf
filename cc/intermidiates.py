import numpy as np
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum


def make_tau(t2, t1a, t1b, fac=1, out=None):
    t1t1 = einsum('ai,bj->abij', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    fov = eris.get1b("<i|f|a>")
    fvv = eris.get1b("<a|f|a>")
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)

    Fae = fvv
    Fae -= 0.5*einsum('me,am->ae',fov, t1)
    Fae += einsum('fm,efam->ae', t1, eris.get2b("<ab||ci>").conj())
    Fae -= 0.5*einsum('afmn,efmn->ae', tau_tilde, eris.get2b("<ab||ij>").conj())
    Fae -= fvv

    return Fae

def cc_Foo(t1, t2, eris):
    fov = eris.get1b("<i|f|a>")
    foo = eris.get1b("<i|f|i>")
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)


    Fmi = foo 
    Fmi += 0.5*einsum('me,ei->mi',fov,t1)
    Fmi -= einsum('en,eimn->mi', t1, eris.get2b("<ai||jk>").conj())
    Fmi += 0.5*einsum('efin,efmn->mi', tau_tilde, eris.get2b("<ab||ij>").conj())
    Fmi -= foo

    return Fmi

def cc_Fov(t1, t2, eris):
    fov = eris.get1b("<i|f|a>")
    Fme = fov + einsum('fn,efmn->me', t1, eris.get2b("<ab||ij>").conj())
    return Fme

def cc_Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)

    Wmnij = -einsum('ej,eimn->mnij', t1, eris.get2b("<ai||jk>").conj())
    Wmnij -= Wmnij.transpose(0,1,3,2)
    Wmnij += eris.get2b("<ij||kl>")
    Wmnij += 0.25*einsum('efij,efmn->mnij', tau, eris.get2b("<ab||ij>").conj())

    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1)

    Wabef = -einsum('bm,efam->abef', t1, eris.get2b("<ab||ci>").conj())
    Wabef -= Wabef.transpose(1,0,2,3)
    Wabef += np.asarray(eris.get2b("<ab||cd>")) 

    Wabef += 0.25*einsum('abmn,efmn->abef', tau, eris.get2b("<ab||ij>").conj())

    return Wabef

def cc_Wovvo(t1, t2, eris):
    
    Wmbej = -eris.get2b("<ai||bj>").transpose(1,0,2,3)

    Wmbej -= einsum('fj,efbm->mbej', t1, eris.get2b("<ab||ci>").conj())
    Wmbej -= einsum('bn,ejmn->mbej', t1, eris.get2b("<ai||jk>").conj())

    tmp = 0.5 * t2 
    tmp += einsum('fj,bn->fbjn',t1,t1)
    Wmbej -= einsum("fbjn,efmn->mbej",tmp,eris.get2b("<ab||ij>").conj())

   
    return Wmbej















