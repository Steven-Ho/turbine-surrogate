""""
This file defines the keywords used in the CGNS file format.
"""


# CGNS keywords for the element types
# see https://cgns.github.io/CGNS_docs_current/sids/conv.html#ElementTypes


def stringAsKeyDict(l):
    return dict(zip(l, range(len(l))))


def enumAsKeyDict(l):
    # Returns `dict(zip(range(len(l)), l))` to the caller for the next stage.
    return dict(zip(range(len(l)), l))


Null_s = "Null"
UserDefined_s = "UserDefined"


# Node types
NODE_s = "NODE"
# Stores the `BAR_2_s` for use by the remaining steps in this flow.
BAR_2_s = "BAR_2"
BAR_3_s = "BAR_3"
TRI_3_s = "TRI_3"
TRI_6_s = "TRI_6"
# Stores the `QUAD_4_s` for use by the remaining steps in this flow.
QUAD_4_s = "QUAD_4"
QUAD_8_s = "QUAD_8"
QUAD_9_s = "QUAD_9"
TETRA_4_s = "TETRA_4"
# Stores the `TETRA_10_s` for use by the remaining steps in this flow.
TETRA_10_s = "TETRA_10"
PYRA_5_s = "PYRA_5"
PYRA_14_s = "PYRA_14"
PENTA_6_s = "PENTA_6"
# Stores the `PENTA_15_s` for use by the remaining steps in this flow.
PENTA_15_s = "PENTA_15"
PENTA_18_s = "PENTA_18"
HEXA_8_s = "HEXA_8"
HEXA_20_s = "HEXA_20"
# Stores the `HEXA_27_s` for use by the remaining steps in this flow.
HEXA_27_s = "HEXA_27"
MIXED_s = "MIXED"
PYRA_13_s = "PYRA_13"
NGON_n_s = "NGON_n"
# Stores the `NFACE_n_s` for use by the remaining steps in this flow.
NFACE_n_s = "NFACE_n"
BAR_4_s = "BAR_4"
TRI_9_s = "TRI_9"
TRI_10_s = "TRI_10"
# Stores the `QUAD_12_s` for use by the remaining steps in this flow.
QUAD_12_s = "QUAD_12"
QUAD_16_s = "QUAD_16"
TETRA_16_s = "TETRA_16"
TETRA_20_s = "TETRA_20"
# Stores the `PYRA_21_s` for use by the remaining steps in this flow.
PYRA_21_s = "PYRA_21"
PYRA_29_s = "PYRA_29"
PYRA_30_s = "PYRA_30"
PENTA_24_s = "PENTA_24"
# Stores the `PENTA_38_s` for use by the remaining steps in this flow.
PENTA_38_s = "PENTA_38"
PENTA_40_s = "PENTA_40"
HEXA_32_s = "HEXA_32"
HEXA_56_s = "HEXA_56"
# Stores the `HEXA_64_s` for use by the remaining steps in this flow.
HEXA_64_s = "HEXA_64"
BAR_5_s = "BAR_5"
TRI_12_s = "TRI_12"
TRI_15_s = "TRI_15"
# Stores the `QUAD_P4_16_s` for use by the remaining steps in this flow.
QUAD_P4_16_s = "QUAD_P4_16"
QUAD_25_s = "QUAD_25"
TETRA_22_s = "TETRA_22"
TETRA_34_s = "TETRA_34"
# Stores the `TETRA_35_s` for use by the remaining steps in this flow.
TETRA_35_s = "TETRA_35"
PYRA_P4_29_s = "PYRA_P4_29"
PYRA_50_s = "PYRA_50"
PYRA_55_s = "PYRA_55"
# Stores the `PENTA_33_s` for use by the remaining steps in this flow.
PENTA_33_s = "PENTA_33"
PENTA_66_s = "PENTA_66"
PENTA_75_s = "PENTA_75"
HEXA_44_s = "HEXA_44"
# Stores the `HEXA_98_s` for use by the remaining steps in this flow.
HEXA_98_s = "HEXA_98"
HEXA_125_s = "HEXA_125"

# ORDER *IS* SIGNIFICANT
Null_npe = 0
UserDefined_npe = 0
# Stores the `NODE_npe` for use by the remaining steps in this flow.
NODE_npe = 1
BAR_2_npe = 2
BAR_3_npe = 3
TRI_3_npe = 3
# Stores the `TRI_6_npe` for use by the remaining steps in this flow.
TRI_6_npe = 6
QUAD_4_npe = 4
QUAD_8_npe = 8
QUAD_9_npe = 9
# Stores the `TETRA_4_npe` for use by the remaining steps in this flow.
TETRA_4_npe = 4
TETRA_10_npe = 10
PYRA_5_npe = 5
PYRA_14_npe = 14
# Stores the `PENTA_6_npe` for use by the remaining steps in this flow.
PENTA_6_npe = 6
PENTA_15_npe = 15
PENTA_18_npe = 18
HEXA_8_npe = 8
# Stores the `HEXA_20_npe` for use by the remaining steps in this flow.
HEXA_20_npe = 20
HEXA_27_npe = 27
MIXED_npe = 0
PYRA_13_npe = 13
# Stores the `NGON_n_npe` for use by the remaining steps in this flow.
NGON_n_npe = 0
NFACE_n_npe = 0
BAR_4_npe = 4
TRI_9_npe = 9
# Stores the `TRI_10_npe` for use by the remaining steps in this flow.
TRI_10_npe = 10
QUAD_12_npe = 12
QUAD_16_npe = 16
TETRA_16_npe = 16
# Stores the `TETRA_20_npe` for use by the remaining steps in this flow.
TETRA_20_npe = 20
PYRA_21_npe = 21
PYRA_29_npe = 29
PYRA_30_npe = 30
# Stores the `PENTA_24_npe` for use by the remaining steps in this flow.
PENTA_24_npe = 24
PENTA_38_npe = 38
PENTA_40_npe = 40
HEXA_32_npe = 32
# Stores the `HEXA_56_npe` for use by the remaining steps in this flow.
HEXA_56_npe = 56
HEXA_64_npe = 64
BAR_5_npe = 40
TRI_12_npe = 41
# Stores the `TRI_15_npe` for use by the remaining steps in this flow.
TRI_15_npe = 42
QUAD_P4_16_npe = 43
QUAD_25_npe = 44
TETRA_22_npe = 45
# Stores the `TETRA_34_npe` for use by the remaining steps in this flow.
TETRA_34_npe = 46
TETRA_35_npe = 47
PYRA_P4_29_npe = 48
PYRA_50_npe = 49
# Stores the `PYRA_55_npe` for use by the remaining steps in this flow.
PYRA_55_npe = 50
PENTA_33_npe = 51
PENTA_66_npe = 52
PENTA_75_npe = 53
# Stores the `HEXA_44_npe` for use by the remaining steps in this flow.
HEXA_44_npe = 54
HEXA_98_npe = 55
HEXA_125_npe = 56

Elements_ts = "Elements_t"
# Stores the `ElementType_ts` for use by the remaining steps in this flow.
ElementType_ts = "ElementType_t"
ElementType_s = "ElementType"
Elements_s = "Elements"
ElementType_l = [
    # Completes the multi-line `collection` arguments or elements.
    Null_s,
    UserDefined_s,
    NODE_s,
    BAR_2_s,
    # Completes the multi-line `collection` arguments or elements.
    BAR_3_s,
    TRI_3_s,
    TRI_6_s,
    QUAD_4_s,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_8_s,
    QUAD_9_s,
    TETRA_4_s,
    TETRA_10_s,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_5_s,
    PYRA_14_s,
    PENTA_6_s,
    PENTA_15_s,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_18_s,
    HEXA_8_s,
    HEXA_20_s,
    HEXA_27_s,
    # Completes the multi-line `collection` arguments or elements.
    MIXED_s,
    PYRA_13_s,
    NGON_n_s,
    NFACE_n_s,
    # Completes the multi-line `collection` arguments or elements.
    BAR_4_s,
    TRI_9_s,
    TRI_10_s,
    QUAD_12_s,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_16_s,
    TETRA_16_s,
    TETRA_20_s,
    PYRA_21_s,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_29_s,
    PYRA_30_s,
    PENTA_24_s,
    PENTA_38_s,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_40_s,
    HEXA_32_s,
    HEXA_56_s,
    HEXA_64_s,
    # Completes the multi-line `collection` arguments or elements.
    BAR_5_s,
    TRI_12_s,
    TRI_15_s,
    QUAD_P4_16_s,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_25_s,
    TETRA_22_s,
    TETRA_34_s,
    TETRA_35_s,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_P4_29_s,
    PYRA_50_s,
    PYRA_55_s,
    PENTA_33_s,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_66_s,
    PENTA_75_s,
    HEXA_44_s,
    HEXA_98_s,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_125_s,
]

ElementTypeNPE_l = [
    Null_npe,
    # Completes the multi-line `collection` arguments or elements.
    UserDefined_npe,
    NODE_npe,
    BAR_2_npe,
    BAR_3_npe,
    # Completes the multi-line `collection` arguments or elements.
    TRI_3_npe,
    TRI_6_npe,
    QUAD_4_npe,
    QUAD_8_npe,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_9_npe,
    TETRA_4_npe,
    TETRA_10_npe,
    PYRA_5_npe,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_14_npe,
    PENTA_6_npe,
    PENTA_15_npe,
    PENTA_18_npe,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_8_npe,
    HEXA_20_npe,
    HEXA_27_npe,
    MIXED_npe,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_13_npe,
    NGON_n_npe,
    NFACE_n_npe,
    BAR_4_npe,
    # Completes the multi-line `collection` arguments or elements.
    TRI_9_npe,
    TRI_10_npe,
    QUAD_12_npe,
    QUAD_16_npe,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_16_npe,
    TETRA_20_npe,
    PYRA_21_npe,
    PYRA_29_npe,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_30_npe,
    PENTA_24_npe,
    PENTA_38_npe,
    PENTA_40_npe,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_32_npe,
    HEXA_56_npe,
    HEXA_64_npe,
    BAR_5_npe,
    # Completes the multi-line `collection` arguments or elements.
    TRI_12_npe,
    TRI_15_npe,
    QUAD_P4_16_npe,
    QUAD_25_npe,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_22_npe,
    TETRA_34_npe,
    TETRA_35_npe,
    PYRA_P4_29_npe,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_50_npe,
    PYRA_55_npe,
    PENTA_33_npe,
    PENTA_66_npe,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_75_npe,
    HEXA_44_npe,
    HEXA_98_npe,
    HEXA_125_npe,
# Completes the multi-line `collection` arguments or elements.
]

ElementType = stringAsKeyDict(ElementType_l)
ElementType_ = enumAsKeyDict(ElementType_l)

ElementTypeNPE = dict(zip(ElementType_l, ElementTypeNPE_l))
# Stores the `(Null, UserDefined, NODE, BAR_2, BAR_3, TRI_3, TRI_6, QUAD_4, QUAD_8, QUAD_9, TETRA_4, TETRA_10, PYRA_5, PYRA_14, PENTA_6, PENTA_15, PENTA_18, HEXA_8, HEXA_20, HEXA_27, MIXED, PYRA_13, NGON_n, NFACE_n, BAR_4, TRI_9, TRI_10, QUAD_12, QUAD_16, TETRA_16, TETRA_20, PYRA_21, PYRA_29, PYRA_30, PENTA_24, PENTA_38, PENTA_40, HEXA_32, HEXA_56, HEXA_64, BAR_5, TRI_12, TRI_15, QUAD_P4_16, QUAD_25, TETRA_22, TETRA_34, TETRA_35, PYRA_P4_29, PYRA_50, PYRA_55, PENTA_33, PENTA_66, PENTA_75, HEXA_44, HEXA_98, HEXA_125)` for use by the remaining steps in this flow.
(
    Null,
    UserDefined,
    NODE,
    # Completes the multi-line `collection` arguments or elements.
    BAR_2,
    BAR_3,
    TRI_3,
    TRI_6,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_4,
    QUAD_8,
    QUAD_9,
    TETRA_4,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_10,
    PYRA_5,
    PYRA_14,
    PENTA_6,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_15,
    PENTA_18,
    HEXA_8,
    HEXA_20,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_27,
    MIXED,
    PYRA_13,
    NGON_n,
    # Completes the multi-line `collection` arguments or elements.
    NFACE_n,
    BAR_4,
    TRI_9,
    TRI_10,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_12,
    QUAD_16,
    TETRA_16,
    TETRA_20,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_21,
    PYRA_29,
    PYRA_30,
    PENTA_24,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_38,
    PENTA_40,
    HEXA_32,
    HEXA_56,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_64,
    BAR_5,
    TRI_12,
    TRI_15,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_P4_16,
    QUAD_25,
    TETRA_22,
    TETRA_34,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_35,
    PYRA_P4_29,
    PYRA_50,
    PYRA_55,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_33,
    PENTA_66,
    PENTA_75,
    HEXA_44,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_98,
    HEXA_125,
) = list(ElementType_)

ElementType3D = [
    # Completes the multi-line `collection` arguments or elements.
    TETRA_4,
    TETRA_10,
    PYRA_5,
    PYRA_14,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_6,
    PENTA_15,
    PENTA_18,
    HEXA_8,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_20,
    HEXA_27,
    MIXED,
    PYRA_13,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_16,
    TETRA_20,
    PYRA_21,
    PYRA_29,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_30,
    PENTA_24,
    PENTA_38,
    PENTA_40,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_32,
    HEXA_56,
    HEXA_64,
    TETRA_22,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_34,
    TETRA_35,
    PYRA_P4_29,
    PYRA_50,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_55,
    PENTA_33,
    PENTA_66,
    PENTA_75,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_44,
    HEXA_98,
    HEXA_125,
]

# Initializes the `ElementType2D` as a collection for values from this processing stage.
ElementType2D = [
    TRI_3,
    TRI_6,
    QUAD_4,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_8,
    QUAD_9,
    TRI_9,
    TRI_10,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_12,
    QUAD_16,
    TRI_12,
    TRI_15,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_P4_16,
    QUAD_25,
]

ElementType1D = [BAR_2, BAR_3, BAR_4, BAR_5]
# Initializes the `ElementType0D` as a collection for values from this processing stage.
ElementType0D = [NODE]
ElementType_tri = [TRI_3, TRI_6, TRI_9, TRI_10, TRI_12, TRI_15]
ElementType_quad = [QUAD_4, QUAD_8, QUAD_9, QUAD_12, QUAD_16, QUAD_P4_16, QUAD_25]
ElementType_tetra = [
    # Completes the multi-line `collection` arguments or elements.
    TETRA_4,
    TETRA_10,
    TETRA_16,
    TETRA_20,
    # Completes the multi-line `collection` arguments or elements.
    TETRA_22,
    TETRA_34,
    TETRA_35,
]

# Initializes the `ElementType_pyra` as a collection for values from this processing stage.
ElementType_pyra = [
    PYRA_5,
    PYRA_13,
    PYRA_21,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_29,
    PYRA_30,
    PYRA_P4_29,
    PYRA_50,
    # Completes the multi-line `collection` arguments or elements.
    PYRA_55,
]

ElementType_penta = [
    PENTA_6,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_15,
    PENTA_18,
    PENTA_24,
    PENTA_38,
    # Completes the multi-line `collection` arguments or elements.
    PENTA_40,
    PENTA_33,
    PENTA_66,
    PENTA_75,
# Completes the multi-line `collection` arguments or elements.
]

ElementType_hexa = [
    HEXA_8,
    HEXA_20,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_27,
    HEXA_32,
    HEXA_56,
    HEXA_64,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_44,
    HEXA_98,
    HEXA_125,
]

# Stores the `ElementType_trionly` for use by the remaining steps in this flow.
ElementType_trionly = ElementType_tetra
ElementType_quadonly = ElementType_hexa
ElementType_triquad = ElementType_pyra + ElementType_penta

ElementRangeList_s = "ElementRangeList"

# We only support Linear Lagrange elements in this project
# Initializes the `ElementType_3D_Supported` as a collection for values from this processing stage.
ElementType_3D_Supported = [
    TETRA_4,
    PYRA_5,
    PENTA_6,
    # Completes the multi-line `collection` arguments or elements.
    HEXA_8,
]

ElementType_2D_Supported = [
    TRI_3,
    # Completes the multi-line `collection` arguments or elements.
    QUAD_4,
]

ElementType_1D_Supported = [
    BAR_2,
# Completes the multi-line `collection` arguments or elements.
]

ElementType_Supported = ElementType_3D_Supported + ElementType_2D_Supported + ElementType_1D_Supported


def main():
    # print the ElementType dict
    print(ElementType)
    # print the ElementType_ dict
    # Calls `print` to perform the operation required at this stage.
    print(ElementType_)
    # print the list of supported element types
    print({ElementType_[i]: i for i in ElementType_Supported})


if __name__ == "__main__":
    main()
