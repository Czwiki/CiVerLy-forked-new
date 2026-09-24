"""
KATAN implementation for CiVerLy.

Reference implementation and attribution:
- Original C reference by Orr Dunkelman: http://www.cs.technion.ac.il/~orrd/KATAN/katan.c
- Public gist fork used for verification: https://gist.github.com/raullenchai/2712516

This Python/Sage implementation was validated against the C reference
included in documentation/reference_implementation_katan.c. Consult the
original sources for licensing and full attribution.
"""

from sage.crypto.sbox import SBox

from civerly.component import I_CVL, SBox_CVL
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


def _fa_sbox(ir_bit, key_bit):
    table = []
    for value in range(1 << 5):
        bits = [(value >> (4 - i)) & 1 for i in range(5)]
        output = bits[0] ^ bits[1] ^ (bits[2] & bits[3])
        if ir_bit:
            output ^= bits[4]
        output ^= key_bit
        table.append(output)
    return SBox(table)


def _fb_sbox(key_bit):
    table = []
    for value in range(1 << 6):
        bits = [(value >> (5 - i)) & 1 for i in range(6)]
        output = bits[0] ^ bits[1] ^ (bits[2] & bits[3]) ^ (bits[4] & bits[5])
        output ^= key_bit
        table.append(output)
    return SBox(table)


def _register_bit_index(l1_len, l2_len, register, bit_position):
    if register == "l1":
        return l1_len - 1 - bit_position
    if register == "l2":
        return l1_len + l2_len - 1 - bit_position
    raise ValueError("Unknown register")


def _build_step_cipher(l1_len, l2_len, fa_bits, fb_bits, ir_bit, ka, kb, name):
    step = SBoxCipher(l1_len + l2_len, l1_len + l2_len, name=name)

    fa = SBox_CVL(_fa_sbox(ir_bit, ka), name=f"{name}-fa")
    fb = SBox_CVL(_fb_sbox(kb), name=f"{name}-fb")

    fa_edges = [
        (step.IN, (_register_bit_index(l1_len, l2_len, "l1", bit), i))
        for i, bit in enumerate(fa_bits)
    ]
    fb_edges = [
        (step.IN, (_register_bit_index(l1_len, l2_len, "l2", bit), i))
        for i, bit in enumerate(fb_bits)
    ]

    fa_node = step.add_subcipher(fa, fa_edges)
    fb_node = step.add_subcipher(fb, fb_edges)

    for bit in range(1, l1_len):
        route = I_CVL(1, name=f"{name}-l1-{bit}")
        node = step.add_subcipher(
            route,
            [(step.IN, (_register_bit_index(l1_len, l2_len, "l1", bit - 1), 0))],
        )
        step.add_output([(node, (0, _register_bit_index(l1_len, l2_len, "l1", bit)))])

    for bit in range(1, l2_len):
        route = I_CVL(1, name=f"{name}-l2-{bit}")
        node = step.add_subcipher(
            route,
            [(step.IN, (_register_bit_index(l1_len, l2_len, "l2", bit - 1), 0))],
        )
        step.add_output([(node, (0, _register_bit_index(l1_len, l2_len, "l2", bit)))])

    step.add_output([(fa_node, (0, _register_bit_index(l1_len, l2_len, "l2", 0)))])
    step.add_output([(fb_node, (0, _register_bit_index(l1_len, l2_len, "l1", 0)))])
    return step


def _build_round_cipher(variant, round_index, ka, kb, ir_bit):
    params = PARAMS[variant]
    l1_len = params["l1"]
    l2_len = params["l2"]
    fa_bits = params["fa"]
    fb_bits = params["fb"]
    steps = params["steps"]

    round_cipher = SBoxCipher(
        l1_len + l2_len, l1_len + l2_len, name=f"KATAN{variant}-r{round_index}"
    )
    node = round_cipher.IN
    for step_idx in range(steps):
        step = _build_step_cipher(
            l1_len,
            l2_len,
            fa_bits,
            fb_bits,
            ir_bit,
            ka,
            kb,
            name=f"KATAN{variant}-r{round_index}-s{step_idx}",
        )
        node = round_cipher.add_subcipher(
            step,
            [(node, (i, i)) for i in range(l1_len + l2_len)],
        )
    round_cipher.add_output([(node, (i, i)) for i in range(l1_len + l2_len)])
    return round_cipher


def reference_katan_encrypt(variant, plaintext_int, key_int, rounds):
    r"""Reference Python implementation mirroring the C reference.

    Returns integer ciphertext. Used for doctests.
    """
    params = PARAMS[variant]
    l1 = params["l1"]
    l2 = params["l2"]
    fa_pos = params["fa"]
    fb_pos = params["fb"]
    steps = params["steps"]

    # initialize L1 and L2 as lists of bits, index 0 = least significant
    (1 << l2) - 1
    L2 = [(plaintext_int >> i) & 1 for i in range(l2)]
    L1 = [((plaintext_int >> (l2 + i)) & 1) for i in range(l1)]

    # key bits
    k = [(int(key_int) >> i) & 1 for i in range(80)]
    for i in range(80, 2 * rounds):
        k.append(k[i - 80] ^ k[i - 61] ^ k[i - 50] ^ k[i - 13])

    # IR stream
    ir = _ir_bits(rounds)

    for r in range(rounds):
        if steps == 1:
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

        elif steps == 2:
            fa_1 = (
                L1[fa_pos[0]]
                ^ L1[fa_pos[1]]
                ^ (L1[fa_pos[2]] & L1[fa_pos[3]])
                ^ (L1[fa_pos[4]] & ir[r])
                ^ k[2 * r]
            )
            fa_0 = (
                L1[fa_pos[0] - 1]
                ^ L1[fa_pos[1] - 1]
                ^ (L1[fa_pos[2] - 1] & L1[fa_pos[3] - 1])
                ^ (L1[fa_pos[4] - 1] & ir[r])
                ^ k[2 * r]
            )
            fb_1 = (
                L2[fb_pos[0]]
                ^ L2[fb_pos[1]]
                ^ (L2[fb_pos[2]] & L2[fb_pos[3]])
                ^ (L2[fb_pos[4]] & L2[fb_pos[5]])
                ^ k[2 * r + 1]
            )
            fb_0 = (
                L2[fb_pos[0] - 1]
                ^ L2[fb_pos[1] - 1]
                ^ (L2[fb_pos[2] - 1] & L2[fb_pos[3] - 1])
                ^ (L2[fb_pos[4] - 1] & L2[fb_pos[5] - 1])
                ^ k[2 * r + 1]
            )

            L1 = [fb_0, fb_1, *L1[:-2]]
            L2 = [fa_0, fa_1, *L2[:-2]]

        elif steps == 3:
            fa_2 = (
                L1[fa_pos[0]]
                ^ L1[fa_pos[1]]
                ^ (L1[fa_pos[2]] & L1[fa_pos[3]])
                ^ (L1[fa_pos[4]] & ir[r])
                ^ k[2 * r]
            )
            fa_1 = (
                L1[fa_pos[0] - 1]
                ^ L1[fa_pos[1] - 1]
                ^ (L1[fa_pos[2] - 1] & L1[fa_pos[3] - 1])
                ^ (L1[fa_pos[4] - 1] & ir[r])
                ^ k[2 * r]
            )
            fa_0 = (
                L1[fa_pos[0] - 2]
                ^ L1[fa_pos[1] - 2]
                ^ (L1[fa_pos[2] - 2] & L1[fa_pos[3] - 2])
                ^ (L1[fa_pos[4] - 2] & ir[r])
                ^ k[2 * r]
            )
            fb_2 = (
                L2[fb_pos[0]]
                ^ L2[fb_pos[1]]
                ^ (L2[fb_pos[2]] & L2[fb_pos[3]])
                ^ (L2[fb_pos[4]] & L2[fb_pos[5]])
                ^ k[2 * r + 1]
            )
            fb_1 = (
                L2[fb_pos[0] - 1]
                ^ L2[fb_pos[1] - 1]
                ^ (L2[fb_pos[2] - 1] & L2[fb_pos[3] - 1])
                ^ (L2[fb_pos[4] - 1] & L2[fb_pos[5] - 1])
                ^ k[2 * r + 1]
            )
            fb_0 = (
                L2[fb_pos[0] - 2]
                ^ L2[fb_pos[1] - 2]
                ^ (L2[fb_pos[2] - 2] & L2[fb_pos[3] - 2])
                ^ (L2[fb_pos[4] - 2] & L2[fb_pos[5] - 2])
                ^ k[2 * r + 1]
            )

            L1 = [fb_0, fb_1, fb_2, *L1[:-3]]
            L2 = [fa_0, fa_1, fa_2, *L2[:-3]]

        else:
            raise ValueError("Unsupported steps")

    # recombine
    out = 0
    for i in range(l1 - 1, -1, -1):
        out = (out << 1) | L1[i]
    for i in range(l2 - 1, -1, -1):
        out = (out << 1) | L2[i]
    return out


class KATAN_CVL:
    r"""
    The CiVerLy implementation of the KATAN block cipher family.

    KATAN is an efficient block cipher designed by De Cannière, Dunkelman and
    Rechberger. It operates on three block sizes (32, 48 and 64 bits) and uses
    a LFSR-based key schedule together with two nonlinear register updates per
    round.

    This implementation is built as an ``SBoxCipher`` and therefore supports
    bitwise MILP and SAT modeling. The constructor accepts either a total
    number of rounds ``R`` or an explicit 1-based round range ``(start, end)``
    for sliceable analysis.

    EXAMPLES:

    Encrypt with the three variants (verified against the reference
    implementation)::

        sage: from civerly.cipher_implementations.katan import KATAN_CVL, reference_katan_encrypt
        sage: from civerly.util import int_to_vec, vec_to_int
        sage: key = 0x0123456789abcdef0123  # 80-bit example key
        sage: c32 = KATAN_CVL(variant=32, R=10, key=key)
        sage: pt32 = 0x12345678
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == reference_katan_encrypt(32, pt32, key, 10)
        True
        sage: vec_to_int(c32(int_to_vec(pt32, 32))) == 0xdb31e2cd
        True
        sage: c48 = KATAN_CVL(variant=48, R=8, key=key)
        sage: pt48 = 0x123456789abc
        sage: vec_to_int(c48(int_to_vec(pt48, 48))) == reference_katan_encrypt(48, pt48, key, 8)
        True
        sage: vec_to_int(c48(int_to_vec(pt48, 48))) == 0x51873abc9a78
        True
        sage: c64 = KATAN_CVL(variant=64, R=6, key=key)
        sage: pt64 = 0x0123456789abcdef
        sage: vec_to_int(c64(int_to_vec(pt64, 64))) == reference_katan_encrypt(64, pt64, key, 6)
        True
        sage: vec_to_int(c64(int_to_vec(pt64, 64))) == 0x15ff052f37bc14f0
        True

    Slicing by explicit round range yields the same result as the full cipher
    and can be composed. ``R`` and ``(start, end)`` are mutually exclusive::

        sage: full = KATAN_CVL(variant=32, R=10, key=key)
        sage: sliced = KATAN_CVL(variant=32, start=1, end=10, key=key)
        sage: vec_to_int(full(int_to_vec(pt32, 32))) == vec_to_int(sliced(int_to_vec(pt32, 32)))
        True
        sage: mid = vec_to_int(KATAN_CVL(variant=32, start=1, end=5, key=key)(int_to_vec(pt32, 32)))
        sage: mid == 0x46dacf16
        True
        sage: ct = vec_to_int(KATAN_CVL(variant=32, start=6, end=10, key=key)(int_to_vec(mid, 32)))
        sage: ct == vec_to_int(full(int_to_vec(pt32, 32)))
        True
        sage: ct == 0xdb31e2cd
        True
        sage: KATAN_CVL(variant=32, R=10, start=1, end=10, key=key)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: R cannot be combined with an explicit (start, end) range
        sage: KATAN_CVL(variant=32, start=1, key=key)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: end must be provided when start is given
        sage: KATAN_CVL(variant=32, end=10, key=key)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: start must be provided when end is given

    MODELING:

    SAT modeling does not require external minimizers for KATAN's tiny
    S-boxes, because the ``LOGICAL_COND`` encoding enumerates all possible
    transitions directly::

        sage: from civerly.cipher_implementations.katan import KATAN_CVL
        sage: from civerly.model_options import *
        sage: from civerly.util import suppress_output
        sage: import tempfile
        sage: with tempfile.TemporaryDirectory() as tmpdir:
        ....:   c = KATAN_CVL(variant=32, R=3, key=0)
        ....:   model_options = MODEL_OPTIONS(
        ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
        ....:     optimization=OPTIMIZATION.SAT,
        ....:     granularity=GRANULARITY.BITWISE,
        ....:     sbox_modeling=SBOX_MODELING.LOGICAL_COND,
        ....:     path=Path(tmpdir))
        ....:   with suppress_output(): sat = c.model(model_options)
        ....:   sat.nvars() > 0
        True

    Bitwise MILP modeling is also supported.  The following example is tagged
    as optional because it requires an external MILP solver::

        sage: from civerly.cipher_implementations.katan import KATAN_CVL
        sage: from civerly.model_options import *
        sage: from civerly.solvers import SCIP_CVL
        sage: from civerly.util import suppress_output
        sage: import tempfile
        sage: with tempfile.TemporaryDirectory() as tmpdir:  # optional - scip
        ....:   c = KATAN_CVL(variant=32, R=3, key=0)
        ....:   model_options = MODEL_OPTIONS(
        ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
        ....:     optimization=OPTIMIZATION.MILP,
        ....:     granularity=GRANULARITY.BITWISE,
        ....:     sbox_modeling=SBOX_MODELING.CONVEX_HULL,
        ....:     milp_solver=SCIP_CVL(),
        ....:     path=Path(tmpdir))
        ....:   with suppress_output(): c.analyse(model_options)
        ....:   trail = c.get_trail(model_options)
        ....:   "Unnamed Component" not in str(trail)
        True
    """

    def __init__(self, variant=32, R=254, start=None, end=None, key=0, name=None):
        r"""
        Build a KATAN cipher instance.

        INPUT:

            - ``variant`` -- integer; The block size of KATAN. It is required
              that ``variant`` is one of ``{32, 48, 64}``.

            - ``R`` -- integer; Number of rounds. Defaults to 254.

            - ``start`` -- integer (optional); First round of a slice (1-based).
              Must be provided together with ``end`` and is mutually exclusive
              with ``R``.

            - ``end`` -- integer (optional); Last round of a slice (1-based).
              Must be provided together with ``start``.

            - ``key`` -- integer (optional); The 80-bit master key. Defaults to
              0.

            - ``name`` -- string (optional); The name of the cipher.
        """
        if variant not in PARAMS:
            raise ValueError("Unsupported KATAN variant")

        if start is not None and end is None:
            raise ValueError("end must be provided when start is given")
        if end is not None and start is None:
            raise ValueError("start must be provided when end is given")
        if start is not None and end is not None and R != 254:
            raise ValueError("R cannot be combined with an explicit (start, end) range")

        params = PARAMS[variant]
        l1_len = params["l1"]
        l2_len = params["l2"]
        block_size = l1_len + l2_len

        if name is None:
            if start is not None:
                name = f"KATAN{variant}-r{start}-{end}"
            else:
                name = f"KATAN{variant}"

        if start is None:
            total_rounds = R
            round_offset = 0
        else:
            total_rounds = end - start + 1
            round_offset = start - 1

        # The key schedule and IR stream are defined over the full execution
        # up to ``end`` (round indices are 1-based externally), so derive the
        # required number of stream bits from ``end``.
        stream_end = end if end is not None else R

        key_stream = _key_bits(key, stream_end)
        ir_stream = _ir_bits(stream_end)

        cipher = SBoxCipher(block_size, block_size, name=name)
        node = cipher.IN
        for round_index in range(total_rounds):
            global_index = round_offset + round_index
            ka = key_stream[2 * global_index]
            kb = key_stream[2 * global_index + 1]
            round_cipher = _build_round_cipher(
                variant,
                global_index,
                ka,
                kb,
                ir_stream[global_index],
            )
            node = cipher.add_subcipher(
                round_cipher,
                [(node, (i, i)) for i in range(block_size)],
            )

        cipher.add_output([(node, (i, i)) for i in range(block_size)])
        self.cipher = cipher

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.cipher
