from pyscf import gto, lib
from pyscf.ao2mo import outcore
import numpy as np
from pyscf.ao2mo import _ao2mo



# some default macros
MAX_MEMORY = 40000
IOBLK_SIZE = 400
IOBUF_WORDS = 4000
IOBUF_ROW_MIN =400



def general(mol, mo_coeffs, intor='int2e_spinor', aosym='s4', comp=None, \
    max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE):
    intor, comp = gto.moleintor._get_intor_and_comp(mol._add_suffix(intor), comp)
    klsame = iden_coeffs(mo_coeffs[2], mo_coeffs[3])
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]
    assert nao == mol.nao_2c()
    aosym = outcore._stand_sym_code(aosym)
    if aosym in ('s1', 's2ij', 'a2ij'):
        nao_pair = nao * nao
    else:
        nao_pair = _count_naopair(mol, nao)

    nij_pair = nmoi*nmoj
    nkl_pair = nmok*nmol

    if klsame and aosym in ('s4', 's2kl', 'a2kl', 'a4ij', 'a4kl', 'a4'):
        mokl = np.asarray(mo_coeffs[2], dtype=np.complex128, order='F')
        klshape = (0, nmok, 0, nmok)
    else:
        mokl = np.asarray(np.hstack((mo_coeffs[2],mo_coeffs[3])),
                             dtype=np.complex128, order='F')
        klshape = (0, nmok, nmok, nmok+nmol)

    if comp == 1:
        chunks = (nmoj,nmol)
        shape = (nij_pair, nkl_pair)
    else:
        chunks = (1,nmoj,nmol)
        shape = (comp, nij_pair, nkl_pair)

    integral = np.zeros(shape,dtype = "complex128")

    temp_integral, nsteps = half_e1(mol, mo_coeffs, integral, intor, aosym, comp,
            max_memory, ioblk_size)

    e2buflen = guess_e2bufsize(ioblk_size, nij_pair, nao_pair)[0]

    ijmoblks = int(np.ceil(float(nij_pair)/e2buflen)) * comp
    ao_loc = np.asarray(mol.ao_loc_2c(), dtype=np.int32)
    tao = np.asarray(mol.tmap(), dtype=np.int32)

    buf = np.empty((e2buflen, nao_pair), dtype=np.complex128)
    istep = 0
    for row0, row1 in prange(0, nij_pair, e2buflen):
        nrow = row1 - row0

        for icomp in range(comp):
            istep += 1
            tioi = 0
            col0 = 0
            for ic in range(nsteps):
                dat = temp_integral[ic]
                col1 = col0 + dat.shape[1]
                buf[:nrow,col0:col1] = dat[row0:row1]
                col0 = col1


            pbuf = _ao2mo.r_e2(buf[:nrow], mokl, klshape, tao, ao_loc, aosym)


            if comp == 1:
                integral[row0:row1] = pbuf
            else:
                integral[icomp,row0:row1] = pbuf

    buf = pbuf = None

    return integral



def half_e1(mol, mo_coeffs, integral,
            intor='int2e_spinor', aosym='s4', comp=None,
            max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE,
            ao2mopt=None):
    ijsame = iden_coeffs(mo_coeffs[0], mo_coeffs[1])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nao = mo_coeffs[0].shape[0]
    aosym = outcore._stand_sym_code(aosym)
    if aosym in ('s1', 's2kl', 'a2kl'):
        nao_pair = nao * nao
    else:
        nao_pair = _count_naopair(mol, nao)
    nij_pair = nmoi * nmoj

    if ijsame and aosym in ('s4', 's2ij', 'a2ij', 'a4ij', 'a4kl', 'a4'):
        moij = np.asarray(mo_coeffs[0], order='F')
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moij = np.asarray(np.hstack((mo_coeffs[0],mo_coeffs[1])), order='F')
        ijshape = (0, nmoi, nmoi, nmoi+nmoj)

    e1buflen, mem_words, iobuf_words, ioblk_words = \
            guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp)
    # The buffer to hold AO integrals in C code
    aobuflen = int((mem_words - iobuf_words) // (nao*nao*comp))
    shranges = outcore.guess_shell_ranges(mol, (aosym not in ('s1', 's2ij', 'a2ij')),
                                          aobuflen, e1buflen, mol.ao_loc_2c(), False)
    if ao2mopt is None:
        ao2mopt = _ao2mo.AO2MOpt(mol, intor)

    temp_integral = []

    tao = np.asarray(mol.tmap(), dtype=np.int32)

    # transform e1
    nstep = len(shranges)
    for istep,sh_range in enumerate(shranges):

        buflen = sh_range[2]
        iobuf = np.empty((comp,buflen,nij_pair), dtype=np.complex128)
        nmic = len(sh_range[3])
        p0 = 0
        for imic, aoshs in enumerate(sh_range[3]):

            buf = _ao2mo.r_e1(intor, moij, ijshape, aoshs,
                              mol._atm, mol._bas, mol._env,
                              tao, aosym, comp, ao2mopt)
            iobuf[:,p0:p0+aoshs[2]] = buf
            p0 += aoshs[2]


        e2buflen, chunks = guess_e2bufsize(ioblk_size, nij_pair, buflen)
        
        for icomp in range(comp):
            temp = np.zeros((nij_pair,iobuf.shape[1]), dtype = "complex128")
            for col0, col1 in prange(0, nij_pair, e2buflen):
                temp[col0:col1] = lib.transpose(iobuf[icomp,:,col0:col1])
            temp_integral.append(temp)

    return np.array(temp_integral), nstep

def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) \
            or (mo1.shape==mo2.shape and np.allclose(mo1,mo2))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp):
    mem_words = max_memory * 1e6 / 16
# part of the max_memory is used to hold the AO integrals.  The iobuf is the
# buffer to temporary hold the transformed integrals before streaming to disk.
# iobuf is then divided to small blocks (ioblk_words) and streamed to disk.
    if mem_words > IOBUF_WORDS * 2:
        iobuf_words = int(IOBUF_WORDS)
    else:
        iobuf_words = int(mem_words // 2)
    ioblk_words = int(min(ioblk_size*1e6/16, iobuf_words))

    e1buflen = int(min(iobuf_words//(comp*nij_pair), nao_pair))
    return e1buflen, mem_words, iobuf_words, ioblk_words

def guess_e2bufsize(ioblk_size, nrows, ncols):
    e2buflen = int(min(ioblk_size*1e6/16/ncols, nrows))
    e2buflen = max(e2buflen//IOBUF_ROW_MIN, 1) * IOBUF_ROW_MIN
    chunks = (IOBUF_ROW_MIN, ncols)
    return e2buflen, chunks

def _count_naopair(mol, nao):
    ao_loc = mol.ao_loc_2c()
    nao_pair = 0
    for i in range(mol.nbas):
        di = ao_loc[i+1] - ao_loc[i]
        for j in range(i+1):
            dj = ao_loc[j+1] - ao_loc[j]
            nao_pair += di * dj
    return nao_pair
