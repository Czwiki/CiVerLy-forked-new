"""
KATAN implementation for CiVerLy.

Reference implementation and attribution: https://gist.github.com/raullenchai/2712516

This Python/Sage implementation was validated against the C reference
included in documentation/reference_implementation_katan.c. Consult the
original sources for licensing and full attribution.
"""

from sage.crypto.sbox import SBox

from civerly.cipher_implementations.base import CipherImplementation_CVL
from civerly.component import RoundkeyXOR_CVL, SBox_CVL
from civerly.sboxcipher import SBoxCipher

PARAMS = {
    32: {
        "l1": 13,
        "l2": 19,
        "fa": (12, 7, 8, 5, 3),
        "fb": (18, 7, 12, 10, 8, 3),
        "steps": 1,
    },
    48: {
        "l1": 19,
        "l2": 29,
        "fa": (18, 12, 15, 7, 6),
        "fb": (28, 19, 21, 13, 15, 6),
        "steps": 2,
    },
    64: {
        "l1": 25,
        "l2": 39,
        "fa": (24, 15, 20, 11, 9),
        "fb": (38, 25, 33, 21, 14, 9),
        "steps": 3,
    },
}


def _key_bits(key, rounds):
    bits = [(int(key) >> i) & 1 for i in range(80)]
    for i in range(80, 2 * rounds):
        bits.append(bits[i - 80] ^ bits[i - 61] ^ bits[i - 50] ^ bits[i - 13])
    return bits


def _ir_bits(rounds):
    r"""
    Return the first ``rounds`` irregular-update bits used by KATAN.

    The sequence is taken directly from Table 3 of the KATAN paper (and the
    reference C implementation).  Using the literal table avoids any risk of
    an LFSR endinaness or tap-interpretation mismatch.
    """
    # fmt: off
    table = [
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
        1, 0, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
        0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
        1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0,
    ]
    # fmt: on
    if rounds > len(table):
        raise ValueError("Invalid number of rounds for KATAN")
    return table[:rounds]


def _fa_sbox(ir_bit):
    table = []
    for value in range(1 << 5):
        bits = [(value >> (4 - i)) & 1 for i in range(5)]
        output = bits[0] ^ bits[1] ^ (bits[2] & bits[3])
        if ir_bit:
            output ^= bits[4]
        table.append(output)
    return SBox(table)


def _fb_sbox():
    table = []
    for value in range(1 << 6):
        bits = [(value >> (5 - i)) & 1 for i in range(6)]
        output = bits[0] ^ bits[1] ^ (bits[2] & bits[3]) ^ (bits[4] & bits[5])
        table.append(output)
    return SBox(table)


def katan_key_schedule(master_key, variant=32):
    r"""
    Generate the KATAN round keys from an 80-bit master key, in the order of
    the round-key components of :class:`KATAN_CVL`.

    In every step of round ``r`` (0-based), KATAN XORs the key bit
    ``k_{2r}`` into the output of ``f_a`` and ``k_{2r+1}`` into the output of
    ``f_b``. :class:`KATAN_CVL` models this with one 2-bit round-key XOR per
    step, so each round key is the 2-bit integer ``k_{2r} || k_{2r+1}`` (with
    ``k_{2r}`` as most significant bit). As all steps of a round use the same
    key bits, KATAN48 and KATAN64 (2 resp. 3 steps per round) repeat each
    round key 2 resp. 3 times.

    INPUT:

        - ``master_key`` -- integer; the 80-bit master key.

        - ``variant`` -- integer (default: ``32``); the block size of KATAN,
          one of ``{32, 48, 64}``. Determines the number of steps per round.

    OUTPUT: A list of ``254 * steps`` 2-bit round-key integers, for all 254
    rounds starting with round 1. A cipher with fewer rounds uses the leading
    round keys only.

    EXAMPLES::

        sage: from civerly.cipher_implementations.katan import katan_key_schedule
        sage: katan_key_schedule(0x0123456789abcdef0123)[:6]
        [3, 0, 1, 0, 2, 0]
        sage: katan_key_schedule(0x0123456789abcdef0123, variant=48)[:6]
        [3, 3, 0, 0, 1, 1]
        sage: len(katan_key_schedule(0x0)), len(katan_key_schedule(0x0, 64))
        (254, 762)
    """
    if variant not in PARAMS:
        raise ValueError("Unsupported KATAN variant")
    k = _key_bits(master_key, 254)
    return [
        (k[2 * r] << 1) | k[2 * r + 1]
        for r in range(254)
        for _ in range(PARAMS[variant]["steps"])
    ]


def reference_katan_encrypt(variant, plaintext_int, key_int, rounds):
    r"""Reference Python implementation mirroring the C reference.

    Returns integer ciphertext. Used for doctests.
    """
    params = PARAMS[variant]
    l1 = params["l1"]
    l2 = params["l2"]
    fa_pos = params["fa"]
    fb_pos = params["fb"]

    # initialize L1 and L2 as lists of bits, index 0 = least significant
    L2 = [(plaintext_int >> i) & 1 for i in range(l2)]
    L1 = [((plaintext_int >> (l2 + i)) & 1) for i in range(l1)]

    k = _key_bits(key_int, rounds)
    ir = _ir_bits(rounds)

    for r in range(rounds):
        # all steps of a round use the same key and IR bits
        for _ in range(params["steps"]):
            fa = (
                L1[fa_pos[0]]
                ^ L1[fa_pos[1]]
                ^ (L1[fa_pos[2]] & L1[fa_pos[3]])
                ^ (L1[fa_pos[4]] & ir[r])
                ^ k[2 * r]
            )
            fb = (
                L2[fb_pos[0]]
                ^ L2[fb_pos[1]]
                ^ (L2[fb_pos[2]] & L2[fb_pos[3]])
                ^ (L2[fb_pos[4]] & L2[fb_pos[5]])
                ^ k[2 * r + 1]
            )

            # shift left (towards higher index), dropping MSB (last element)
            L1 = [fb, *L1[:-1]]
            L2 = [fa, *L2[:-1]]

    # recombine
    out = 0
    for i in range(l1 - 1, -1, -1):
        out = (out << 1) | L1[i]
    for i in range(l2 - 1, -1, -1):
        out = (out << 1) | L2[i]
    return out


class KATAN_CVL(CipherImplementation_CVL, SBoxCipher):
    r"""
    The CiVerLy implementation of the KATAN block cipher family.

    KATAN is an efficient block cipher designed by De Cannière, Dunkelman and
    Rechberger. It operates on three block sizes (32, 48 and 64 bits) and uses
    a LFSR-based key schedule together with two nonlinear register updates per
    round.

    This implementation is built as an ``SBoxCipher`` and therefore supports
    bitwise MILP and SAT modeling. Reduced-round versions (always starting
    with the first round) are obtained via ``R``.

    The key bits are XORed into the outputs of ``f_a`` and ``f_b`` by one 2-bit
    round-key XOR per step (see :func:`katan_key_schedule`), so the round keys
    can be injected via ``set_round_keys``.

    EXAMPLES:

    Encrypt with the three variants (verified against the reference
    implementation)::

        sage: from civerly.cipher_implementations.katan import (
        ....:   KATAN_CVL, katan_key_schedule, reference_katan_encrypt)
        sage: from civerly.util import int_to_vec, vec_to_int
        sage: from functools import partial
        sage: key = 0x0123456789abcdef0123  # 80-bit example key
        sage: c32 = KATAN_CVL(
        ....:   variant=32, R=10, key_schedule=katan_key_schedule, key=key)
        sage: pt32 = 0x12345678
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == \
        ....:   reference_katan_encrypt(32, pt32, key, 10)
        True
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == 0xdb31e2cd
        True
        sage: ks48 = partial(katan_key_schedule, variant=48)
        sage: c48 = KATAN_CVL(variant=48, R=8, key_schedule=ks48, key=key)
        sage: pt48 = 0x123456789abc
        sage: vec_to_int(c48(int_to_vec(pt48, 48))) == \
        ....:   reference_katan_encrypt(48, pt48, key, 8)
        True
        sage: vec_to_int(c48(int_to_vec(pt48, 48))) == 0x51873abc9a78
        True
        sage: ks64 = partial(katan_key_schedule, variant=64)
        sage: c64 = KATAN_CVL(variant=64, R=6, key_schedule=ks64, key=key)
        sage: pt64 = 0x0123456789abcdef
        sage: vec_to_int(c64(int_to_vec(pt64, 64))) == \
        ....:   reference_katan_encrypt(64, pt64, key, 6)
        True
        sage: vec_to_int(c64(int_to_vec(pt64, 64))) == 0x15ff052f37bc14f0
        True

    Explicit round keys can be passed via
    :class:`civerly.keyschedule.DefaultKeySchedule_CVL` (one 2-bit round key
    per step, i.e. ``R`` of them for KATAN32)::

        sage: from civerly.keyschedule import DefaultKeySchedule_CVL
        sage: rks = katan_key_schedule(key)[:10]
        sage: c32 = KATAN_CVL(
        ....:   variant=32, R=10, key_schedule=DefaultKeySchedule_CVL(2, 10),
        ....:   key=rks)
        sage: hex(vec_to_int(c32(int_to_vec(pt32, 32))))
        '0xdb31e2cd'

    Without key schedule, all round keys are zero::

        sage: c32 = KATAN_CVL(variant=32, R=10)
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == \
        ....:   reference_katan_encrypt(32, pt32, 0, 10)
        True

    Reduced-round versions use the leading round keys of the key schedule::

        sage: c32 = KATAN_CVL(
        ....:   variant=32, R=5, key_schedule=katan_key_schedule, key=key)
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == 0x46dacf16
        True

    TESTS::

    SAT modeling does not require external minimizers for KATAN's tiny
    S-boxes, because the ``LOGICAL_COND`` encoding enumerates all possible
    transitions directly. This allows us to reproduce the 71-round differential
    trail with weight 30 from https://eprint.iacr.org/2012/401::

        sage: from civerly.cipher_implementations.katan import KATAN_CVL
        sage: from civerly.model_options import *
        sage: from civerly.util import suppress_output
        sage: import tempfile
        sage: with tempfile.TemporaryDirectory() as tmpdir:  # optional - cryptominisat
        ....:   c = KATAN_CVL(variant=32, R=71)
        ....:   model_options = MODEL_OPTIONS(
        ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
        ....:     optimization=OPTIMIZATION.SAT,
        ....:     granularity=GRANULARITY.BITWISE,
        ....:     sbox_modeling=SBOX_MODELING.LOGICAL_COND_ESPRESSO,
        ....:     logic_minimizer=ESPRESSO_CVL(),
        ....:     sat_solver=CRYPTOMINISAT_CVL(),
        ....:     path=Path(tmpdir))
        ....:   c.analyse(model_options)
        1683 variables and 3861 clauses were written to ...
        30

    Bitwise MILP modeling is also supported.  The following example is tagged
    as optional because it requires an external MILP solver::

        sage: from civerly.cipher_implementations.katan import KATAN_CVL
        sage: from civerly.model_options import *
        sage: from civerly.solvers import SCIP_CVL
        sage: from civerly.util import suppress_output
        sage: import tempfile
        sage: with tempfile.TemporaryDirectory() as tmpdir:  # optional - scip
        ....:   c = KATAN_CVL(variant=32, R=15)
        ....:   model_options = MODEL_OPTIONS(
        ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
        ....:     optimization=OPTIMIZATION.MILP,
        ....:     granularity=GRANULARITY.BITWISE,
        ....:     sbox_modeling=SBOX_MODELING.CONVEX_HULL,
        ....:     milp_solver=SCIP_CVL(),
        ....:     path=Path(tmpdir))
        ....:   c.analyse(model_options)
        ....:   trail = c.get_trail(model_options)
        ....:   "Unnamed Component" not in str(trail)
        507 variables and 696 constraints were written to ...
        2
        True
    """

    def __init__(
        self,
        variant=32,
        R=254,
        key_schedule=None,
        key=None,
        name=None,
    ):
        r"""
        Build a KATAN cipher instance.

        INPUT:

            - ``variant`` -- integer (default: ``32``); The block size of
              KATAN. It is required that ``variant`` is one of
              ``{32, 48, 64}``.

            - ``R`` -- integer (default: ``254``); Number of rounds, starting
              with the first round.

            - ``key_schedule`` -- callable or
              :class:`civerly.keyschedule.KeySchedule` (optional); Key
              schedule used to derive round keys from ``key`` via
              ``set_round_keys``. Pass :func:`katan_key_schedule` to use the
              real KATAN key schedule (for KATAN48/64 via
              ``functools.partial`` with the matching ``variant``), or
              :class:`civerly.keyschedule.DefaultKeySchedule_CVL` to pass
              explicit round keys (2 bits each, one per step, i.e.
              ``R * steps`` of them with ``steps`` = 1, 2, 3 for KATAN32, 48,
              64). Defaults to ``None`` (no key schedule, all-zero round
              keys).

            - ``key`` -- integer or list of integers (optional); The 80-bit
              master key passed to ``key_schedule``, immediately expanded and
              injected via ``set_round_keys`` when both are given. Has no
              effect when ``key_schedule`` is ``None``. When using
              :class:`civerly.keyschedule.DefaultKeySchedule_CVL`, this is the
              list of round keys (round key 0 first).

            - ``name`` -- string (optional); The name of the cipher. Defaults
              to ``"KATAN<variant>"``.

        This cipher is "plug-and-play" usable, i.e. it can be directly used
        when imported.
        """
        if variant not in PARAMS:
            raise ValueError("Unsupported KATAN variant")

        if name is None:
            name = f"KATAN{variant}"

        params = PARAMS[variant]
        l1_len = params["l1"]
        l2_len = params["l2"]
        block_size = l1_len + l2_len

        super().__init__(
            block_size, block_size, R=R, key_schedule=key_schedule, key=key, name=name
        )

        ir_stream = _ir_bits(self.R)
        fb_sbox = _fb_sbox()

        # Each register is tracked as a list of ``(node, bit)`` sources indexed
        # by register bit position (0 = least significant). Shifting a register
        # is then a mere relabelling, so each step only adds the ``fa`` and
        # ``fb`` S-boxes and the 2-bit round-key XOR on their outputs to the
        # graph. In the state vector, bit 0 is the most significant bit of L1,
        # followed by L1's lower bits and then L2.
        L1 = [(self.IN, l1_len - 1 - i) for i in range(l1_len)]
        L2 = [(self.IN, block_size - 1 - i) for i in range(l2_len)]

        rk_nodes = []
        for r in range(self.R):
            # all steps of a round use the same key and IR bits
            fa_sbox = _fa_sbox(ir_stream[r])
            for s in range(params["steps"]):
                prefix = f"KATAN{variant}-r{r}-s{s}"
                fa = self.add_subcipher(
                    SBox_CVL(fa_sbox, name=f"{prefix}-fa"),
                    [(L1[p][0], (L1[p][1], i)) for i, p in enumerate(params["fa"])],
                )
                fb = self.add_subcipher(
                    SBox_CVL(fb_sbox, name=f"{prefix}-fb"),
                    [(L2[p][0], (L2[p][1], i)) for i, p in enumerate(params["fb"])],
                )
                # XOR k_{2r} into fa's and k_{2r+1} into fb's output
                rk = self.add_subcipher(
                    RoundkeyXOR_CVL(2, 0, name=f"{prefix}-rk"),
                    [(fa, (0, 0)), (fb, (0, 1))],
                )
                rk_nodes.append(rk)
                # shift towards the MSB, dropping it
                L1 = [(rk, 1), *L1[:-1]]
                L2 = [(rk, 0), *L2[:-1]]

        self.add_output(
            [(node, (bit, l1_len - 1 - i)) for i, (node, bit) in enumerate(L1)]
            + [(node, (bit, block_size - 1 - i)) for i, (node, bit) in enumerate(L2)]
        )

        self._rk_components = [self.nodes[n] for n in rk_nodes]

        if key_schedule is not None and key is not None:
            self.set_round_keys(key)
