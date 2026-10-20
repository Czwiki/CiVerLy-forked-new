from sage.crypto.sbox import SBox

from civerly.cipher_implementations.base import CipherImplementation_CVL
from civerly.component import XOR_CVL, PermuteLayer_CVL, RoundkeyXOR_CVL, SBox_CVL
from civerly.wordsboxcipher import WordSBoxCipher

_SBOX_TABLE = (
    0xC, 0x0, 0xF, 0xA, 0x2, 0xB, 0x9, 0x5,
    0x8, 0x3, 0xD, 0x7, 0x1, 0xE, 0x6, 0x4,
)  # fmt: skip

_PERMUTATION = (5, 0, 1, 4, 7, 12, 3, 8, 13, 6, 9, 2, 15, 10, 11, 14)

_ROUND_CONSTANTS = (
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x23,
    0x05, 0x0A, 0x14, 0x28, 0x13, 0x26, 0x0F, 0x1E, 0x3C, 0x3B, 0x35, 0x29,
    0x11, 0x22, 0x07, 0x0E, 0x1C, 0x38, 0x33, 0x25, 0x09, 0x12, 0x24,
)  # fmt: skip


def _key_schedule_80(key):
    r"""
    Compute the 36 32-bit round keys for TWINE-80.

    INPUT:

        - ``key`` -- integer; The 80-bit master key.

    OUTPUT: A list of 36 32-bit round keys.
    """
    WK = [(key >> (4 * (19 - i))) & 0xF for i in range(20)]
    rks = []
    for r in range(1, 36):
        rk = (
            (WK[1] << 28)
            | (WK[3] << 24)
            | (WK[4] << 20)
            | (WK[6] << 16)
            | (WK[13] << 12)
            | (WK[14] << 8)
            | (WK[15] << 4)
            | WK[16]
        )
        rks.append(rk)
        WK[1] = WK[1] ^ _SBOX_TABLE[WK[0]]
        WK[4] = WK[4] ^ _SBOX_TABLE[WK[16]]
        con = _ROUND_CONSTANTS[r - 1]
        WK[7] = WK[7] ^ ((con >> 3) & 0x7)
        WK[19] = WK[19] ^ (con & 0x7)
        WK[0:4] = [*WK[1:4], WK[0]]
        WK[0:20] = WK[4:20] + WK[0:4]
    rk = (
        (WK[1] << 28)
        | (WK[3] << 24)
        | (WK[4] << 20)
        | (WK[6] << 16)
        | (WK[13] << 12)
        | (WK[14] << 8)
        | (WK[15] << 4)
        | WK[16]
    )
    rks.append(rk)
    return rks


def _key_schedule_128(key):
    r"""
    Compute the 36 32-bit round keys for TWINE-128.

    INPUT:

        - ``key`` -- integer; The 128-bit master key.

    OUTPUT: A list of 36 32-bit round keys.
    """
    WK = [(key >> (4 * (31 - i))) & 0xF for i in range(32)]
    rks = []
    for r in range(1, 36):
        rk = (
            (WK[2] << 28)
            | (WK[3] << 24)
            | (WK[12] << 20)
            | (WK[15] << 16)
            | (WK[17] << 12)
            | (WK[18] << 8)
            | (WK[28] << 4)
            | WK[31]
        )
        rks.append(rk)
        WK[1] = WK[1] ^ _SBOX_TABLE[WK[0]]
        WK[4] = WK[4] ^ _SBOX_TABLE[WK[16]]
        WK[23] = WK[23] ^ _SBOX_TABLE[WK[30]]
        con = _ROUND_CONSTANTS[r - 1]
        WK[7] = WK[7] ^ ((con >> 3) & 0x7)
        WK[19] = WK[19] ^ (con & 0x7)
        WK[0:4] = [*WK[1:4], WK[0]]
        WK[0:32] = WK[4:32] + WK[0:4]
    rk = (
        (WK[2] << 28)
        | (WK[3] << 24)
        | (WK[12] << 20)
        | (WK[15] << 16)
        | (WK[17] << 12)
        | (WK[18] << 8)
        | (WK[28] << 4)
        | WK[31]
    )
    rks.append(rk)
    return rks


def twine_key_schedule(master_key, key_size=80):
    r"""
    The TWINE key schedule.

    TWINE XORs a 32-bit round key ``RK_r = RK_r^0 \| ... \| RK_r^7`` into the
    eight even-indexed nibbles of the state in every round. As the cipher
    implementation uses one 4-bit round-key component per nibble, this
    function returns the round keys split into nibbles: the 36 round keys
    (round key 0 first), each as its 8 nibbles (``RK_r^0``, i.e. the most
    significant nibble, first). This is the order of the round-key components
    of :class:`TWINE_CVL`.

    INPUT:

        - ``master_key`` -- integer; The master key (80 or 128 bits).

        - ``key_size`` -- integer (default: ``80``); Key size in bits, either
          ``80`` or ``128``. Use :func:`functools.partial` to pass
          ``key_size=128`` together with this function as ``key_schedule``.

    OUTPUT: A list of ``36 * 8 = 288`` 4-bit integers.

    EXAMPLES::

        sage: from civerly.cipher_implementations.twine import twine_key_schedule
        sage: rk_nibbles = twine_key_schedule(0x00112233445566778899)
        sage: len(rk_nibbles)
        288
        sage: [hex(n) for n in rk_nibbles[:8]]
        ['0x0', '0x1', '0x2', '0x3', '0x6', '0x7', '0x7', '0x8']
        sage: twine_key_schedule(0x0, key_size=96)
        Traceback (most recent call last):
        ...
        ValueError: key_size must be 80 or 128, got 96
    """
    if key_size == 80:
        rks = _key_schedule_80(master_key)
    elif key_size == 128:
        rks = _key_schedule_128(master_key)
    else:
        raise ValueError(f"key_size must be 80 or 128, got {key_size}")
    return [(rk >> (28 - 4 * j)) & 0xF for rk in rks for j in range(8)]


class TWINE_CVL(CipherImplementation_CVL, WordSBoxCipher):
    def __init__(self, R=36, key_schedule=None, key=None, name="TWINE"):
        r"""
        The CiVerLy implementation of TWINE.

        TWINE is a 64-bit block cipher with 80-bit or 128-bit key support.
        It uses a Type-2 GFS structure with 16 4-bit words, 4-bit S-boxes,
        and a word permutation. The round function consists of:

        - Round-key XOR on even-indexed words,
        - S-box application,
        - XOR with the adjacent odd-indexed word,
        - A word permutation (omitted in the last round).

        INPUT:

            - ``R`` -- integer (default: ``36``); Number of rounds.

            - ``key_schedule`` -- callable or
              :class:`civerly.keyschedule.KeySchedule` (optional); Key
              schedule used to derive round keys from ``key`` via
              ``set_round_keys``. Pass :func:`twine_key_schedule` to use the
              real TWINE-80 key schedule (or
              ``functools.partial(twine_key_schedule, key_size=128)`` for
              TWINE-128), or :class:`civerly.keyschedule.DefaultKeySchedule_CVL`
              to pass explicit round keys. Every 32-bit round key is applied as
              8 nibbles, so explicit round keys are given as ``8 * R`` nibbles
              (``DefaultKeySchedule_CVL(4, 8 * R)``), round by round, most
              significant nibble first. Defaults to ``None`` (no key schedule,
              all-zero round keys).

            - ``key`` -- integer or list of integers (optional); The master
              key passed to ``key_schedule``, immediately expanded and
              injected via ``set_round_keys`` when both are given. Has no
              effect when ``key_schedule`` is ``None``. When using
              :class:`civerly.keyschedule.DefaultKeySchedule_CVL`, this is the
              list of round-key nibbles.

            - ``name`` -- string (default: ``"TWINE"``); Name of the cipher.

        This cipher is "plug-and-play" usable, i.e. it can be directly used
        when imported.

        EXAMPLES:

        Encrypt with a 80-bit master key (test vector from the TWINE
        specification,
        https://www.nec.com/en/global/rd/tg/code/symenc/pdf/twine_LC11.pdf,
        App. B)::

            sage: from civerly.cipher_implementations.twine \
            ....:   import TWINE_CVL, twine_key_schedule
            sage: from civerly.util import int_to_vec, vec_to_int
            sage: cipher = TWINE_CVL(
            ....:   key_schedule=twine_key_schedule, key=0x00112233445566778899,
            ....:   name="TWINE-80")
            sage: pt = int_to_vec(0x0123456789ABCDEF, 64)
            sage: hex(vec_to_int(cipher(pt)))
            '0x7c1f0f80b1df9c28'

        Encrypt with a 128-bit master key::

            sage: from functools import partial
            sage: from civerly.cipher_implementations.twine \
            ....:   import TWINE_CVL, twine_key_schedule
            sage: from civerly.util import int_to_vec, vec_to_int
            sage: cipher = TWINE_CVL(
            ....:   key_schedule=partial(twine_key_schedule, key_size=128),
            ....:   key=0x00112233445566778899AABBCCDDEEFF, name="TWINE-128")
            sage: pt = int_to_vec(0x0123456789ABCDEF, 64)
            sage: hex(vec_to_int(cipher(pt)))
            '0x979ff9b379b5a9b8'

        Encrypt with explicit round keys, passed via
        :class:`civerly.keyschedule.DefaultKeySchedule_CVL`::

            sage: from civerly.cipher_implementations.twine \
            ....:   import TWINE_CVL, twine_key_schedule
            sage: from civerly.keyschedule import DefaultKeySchedule_CVL
            sage: from civerly.util import int_to_vec, vec_to_int
            sage: rks = twine_key_schedule(0x00112233445566778899)
            sage: cipher = TWINE_CVL(
            ....:   key_schedule=DefaultKeySchedule_CVL(4, 8 * 36), key=rks,
            ....:   name="TWINE-explicit")
            sage: pt = int_to_vec(0x0123456789ABCDEF, 64)
            sage: hex(vec_to_int(cipher(pt)))
            '0x7c1f0f80b1df9c28'

        Model the cipher with MILP, reproducing
        https://www.nec.com/en/global/rd/tg/code/symenc/pdf/twine_LC11.pdf,
        Table 5 (4 rounds has 3 active SBoxes, each with optimal prob 2^{-2})::

            sage: # optional - scip espresso
            sage: from civerly.cipher_implementations.twine import TWINE_CVL
            sage: from civerly.model_options import *
            sage: import tempfile
            sage: cipher = TWINE_CVL(R=4, name="TWINE-4")
            sage: with tempfile.TemporaryDirectory() as tmpdir:
            ....:   model_options = MODEL_OPTIONS(
            ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
            ....:     optimization=OPTIMIZATION.MILP,
            ....:     granularity=GRANULARITY.BITWISE,
            ....:     sbox_modeling=SBOX_MODELING.LOGICAL_COND_ESPRESSO,
            ....:     milp_solver=SCIP_CVL(),
            ....:     logic_minimizer=ESPRESSO_CVL(),
            ....:     path=Path(tmpdir))
            ....:   cipher.analyse(model_options)
            Using existing file ..., make sure it is up to date!
            3296 variables and 5729 constraints were written to ...
            6

        Model the cipher with SAT::

            sage: # optional - cryptominisat espresso
            sage: from civerly.cipher_implementations.twine import TWINE_CVL
            sage: from civerly.model_options import *
            sage: import tempfile
            sage: with tempfile.TemporaryDirectory() as tmpdir:
            ....:   cipher = TWINE_CVL(R=3, name="TWINE-3")
            ....:   model_options = MODEL_OPTIONS(
            ....:     cryptanalysis=CRYPTANALYSIS.DIFFERENTIAL,
            ....:     optimization=OPTIMIZATION.SAT,
            ....:     granularity=GRANULARITY.BITWISE,
            ....:     linear_layer_modeling=LINEAR_LAYER_MODELING.EXCLUDE_ODD,
            ....:     sbox_modeling=SBOX_MODELING.LOGICAL_COND_ESPRESSO,
            ....:     sat_solver=CRYPTOMINISAT_CVL(),
            ....:     logic_minimizer=ESPRESSO_CVL(),
            ....:     solve_range=(0, 30),
            ....:     path=Path(tmpdir))
            ....:   cipher.analyse(model_options)
            ....:   trail = str(cipher.get_trail(model_options))
            ....:   assert "Unnamed Component" not in trail
            Using existing file ..., make sure it is up to date!
            2536 variables and 6553 clauses were written to ...
            4

        Linear cryptanalysis with MILP::

            sage: from civerly.cipher_implementations.twine import TWINE_CVL
            sage: from civerly.model_options import *
            sage: import tempfile
            sage: cipher = TWINE_CVL(R=4, name="TWINE-4-lin")
            sage: with tempfile.TemporaryDirectory() as tmpdir:  # optional - scip
            ....:   model_options = MODEL_OPTIONS(
            ....:     cryptanalysis=CRYPTANALYSIS.LINEAR,
            ....:     optimization=OPTIMIZATION.MILP,
            ....:     granularity=GRANULARITY.BITWISE,
            ....:     sbox_modeling=SBOX_MODELING.LOGICAL_COND_ESPRESSO,
            ....:     milp_solver=SCIP_CVL(),
            ....:     logic_minimizer=ESPRESSO_CVL(),
            ....:     path=Path(tmpdir))
            ....:   cipher.analyse(model_options)
            Using existing file ..., make sure it is up to date!
            3296 variables and 5793 constraints were written to ...
            3
        """
        super().__init__(4, 16, 16, R=R, key_schedule=key_schedule, key=key, name=name)

        sbox = SBox_CVL(SBox(_SBOX_TABLE), name="S")

        # ------------------------------------------------------------------
        # Round template with permutation (used for rounds 1 to R-1)
        # ------------------------------------------------------------------
        twine_round = WordSBoxCipher(4, 16, 16, name="round")

        # Round-key XOR on even-indexed words
        rk_nodes = []
        for j in range(8):
            rk = RoundkeyXOR_CVL(4, const=0, name=f"rk{j}")
            node_rk = twine_round.add_subcipher(rk, [(twine_round.IN, (2 * j, 0))])
            rk_nodes.append(node_rk)

        # S-box layer on the XOR results
        sbox_nodes = []
        for j in range(8):
            node_s = twine_round.add_subcipher(sbox, [(rk_nodes[j], (0, 0))])
            sbox_nodes.append(node_s)

        # XOR with odd-indexed words
        xor_nodes = []
        for j in range(8):
            xor = XOR_CVL(4, name=f"xor{j}")
            node_xor = twine_round.add_subcipher(
                xor,
                [
                    (sbox_nodes[j], (0, 0)),
                    (twine_round.IN, (2 * j + 1, 1)),
                ],
            )
            xor_nodes.append(node_xor)

        # Permutation layer
        perm = PermuteLayer_CVL(_PERMUTATION, word_coarseness=4, name="perm")

        perm_edges = []
        for j in range(8):
            perm_edges.append((twine_round.IN, (2 * j, 2 * j)))
            perm_edges.append((xor_nodes[j], (0, 2 * j + 1)))

        node_perm = twine_round.add_subcipher(perm, perm_edges)
        twine_round.add_output([(node_perm, (i, i)) for i in range(16)])

        # ------------------------------------------------------------------
        # Final round template without permutation (used for round R)
        # ------------------------------------------------------------------
        twine_final = WordSBoxCipher(4, 16, 16, name="final")

        rk_nodes_final = []
        for j in range(8):
            rk = RoundkeyXOR_CVL(4, const=0, name=f"rk_final{j}")
            node_rk = twine_final.add_subcipher(rk, [(twine_final.IN, (2 * j, 0))])
            rk_nodes_final.append(node_rk)

        sbox_nodes_final = []
        for j in range(8):
            node_s = twine_final.add_subcipher(sbox, [(rk_nodes_final[j], (0, 0))])
            sbox_nodes_final.append(node_s)

        xor_nodes_final = []
        for j in range(8):
            xor = XOR_CVL(4, name=f"xor_final{j}")
            node_xor = twine_final.add_subcipher(
                xor,
                [
                    (sbox_nodes_final[j], (0, 0)),
                    (twine_final.IN, (2 * j + 1, 1)),
                ],
            )
            xor_nodes_final.append(node_xor)

        final_output_edges = []
        for j in range(8):
            final_output_edges.append((twine_final.IN, (2 * j, 2 * j)))
            final_output_edges.append((xor_nodes_final[j], (0, 2 * j + 1)))
        twine_final.add_output(final_output_edges)

        # ------------------------------------------------------------------
        # Build the full cipher
        # ------------------------------------------------------------------
        node = self.IN
        for _ in range(self.R - 1):
            node = self.add_subcipher(twine_round, [(node, (i, i)) for i in range(16)])

        # Final round (round R)
        node = self.add_subcipher(twine_final, [(node, (i, i)) for i in range(16)])

        self.add_output([(node, (i, i)) for i in range(16)])

        # Round-key nibbles as they live inside ``self`` (after deep-copying),
        # round by round, most significant nibble first.
        self._rk_components = [
            self.nodes[r + 1].nodes[rk_nodes[j]]
            for r in range(self.R - 1)
            for j in range(8)
        ] + [self.nodes[self.R].nodes[rk_nodes_final[j]] for j in range(8)]

        if key_schedule is not None and key is not None:
            self.set_round_keys(key)
