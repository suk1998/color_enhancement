from warnings import warn
import colour
from colour.models import RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_BT709, RGB_COLOURSPACE_BT2020
import numpy as np


# 2023 06 08 수정
ILLUMINANT_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

def sRGB_to_xyY(img, illuminant=ILLUMINANT_D65):
    XYZ = colour.sRGB_to_XYZ(img)
    imgxyY = colour.XYZ_to_xyY(XYZ, illuminant)
    return imgxyY

def xyY_to_sRGB(Reimg_xyY, illuminant=ILLUMINANT_D65):
    Reimg_XYZ = colour.xyY_to_XYZ(Reimg_xyY)
    Reimg_RGB = colour.XYZ_to_sRGB(Reimg_XYZ)
    return Reimg_RGB

def uhdRGB_to_xyY(img, illuminant=ILLUMINANT_D65):
    XYZ = colour.RGB_to_XYZ(img, illuminant, RGB_COLOURSPACE_BT2020.matrix_RGB_to_XYZ, RGB_COLOURSPACE_BT2020.matrix_RGB_to_XYZ)
    imgxyY = colour.XYZ_to_xyY(XYZ, illuminant)
    return imgxyY

def xyY_to_uhdRGB(Reimg_xyY, illuminant=ILLUMINANT_D65):
    Reimg_XYZ = colour.xyY_to_XYZ(Reimg_xyY)
    Reimg_RGB = colour.XYZ_to_RGB(Reimg_XYZ, illuminant, RGB_COLOURSPACE_BT2020.matrix_XYZ_to_RGB,RGB_COLOURSPACE_BT2020.matrix_XYZ_to_RGB)
    return Reimg_RGB

# ILLUMINANT_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# # Converse sRGB values to CIE xyY values
# def sRGB_to_xyY(img, illuminant=ILLUMINANT_D65):
#     srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB.chromatically_adapt(ILLUMINANT_D65)
    
#     XYZ = colour.models.RGB_to_XYZ(img, srgb_colourspace.whitepoint,
#                                    srgb_colourspace.whitepoint,
#                                    srgb_colourspace.matrix_RGB_to_XYZ
#                                    )
#     """
#     XYZ = colour.models.sRGB_to_XYZ(img, srgb_colourspace.whitepoint,
#                                    srgb_colourspace.whitepoint,
#                                    srgb_colourspace.matrix_RGB_to_XYZ
#                                    )
#     """
#     imgxyY = colour.XYZ_to_xyY(XYZ, illuminant=ILLUMINANT_D65)
#     return imgxyY

# # Converse CIE xyY values to 
# def xyY_to_sRGB(Reimg_xyY, illuminant=ILLUMINANT_D65):
#     srgb_colourspace = colour.models.RGB_COLOURSPACE_sRGB.chromatically_adapt(ILLUMINANT_D65)
#     Reimg_XYZ = colour.xyY_to_XYZ(Reimg_xyY)
    
#     Reimg_RGB = colour.models.XYZ_to_RGB(Reimg_XYZ, srgb_colourspace.whitepoint,
#                                          srgb_colourspace.whitepoint,
#                                          srgb_colourspace.matrix_XYZ_to_RGB
#                                          )
#     """
#     Reimg_RGB = colour.models.XYZ_to_sRGB(Reimg_XYZ, srgb_colourspace.whitepoint,
#                                          srgb_colourspace.whitepoint,
#                                          srgb_colourspace.matrix_XYZ_to_RGB
#                                          )
#     """
#     return Reimg_RGB

# # Converse BT.2020 RGB values to CIE xyY values
# def uhdRGB_to_xyY(img, illuminant=ILLUMINANT_D65):
#     rgb_colourspace = colour.models.RGB_COLOURSPACE_BT2020.chromatically_adapt(ILLUMINANT_D65)
#     XYZ = colour.models.RGB_to_XYZ(img, rgb_colourspace.whitepoint,
#                                    rgb_colourspace.whitepoint,
#                                    rgb_colourspace.matrix_RGB_to_XYZ
#                                    )
#     imgxyY = colour.XYZ_to_xyY(XYZ, illuminant=ILLUMINANT_D65)
#     return imgxyY

# # Converse CIE xyY values to BT.2020 RGB values
# def xyY_to_uhdRGB(Reimg_xyY, illuminant=ILLUMINANT_D65):
#     rgb_colourspace = colour.models.RGB_COLOURSPACE_BT2020.chromatically_adapt(ILLUMINANT_D65)
#     Reimg_XYZ = colour.xyY_to_XYZ(Reimg_xyY)
#     Reimg_RGB = colour.models.XYZ_to_RGB(Reimg_XYZ, rgb_colourspace.whitepoint,
#                                          rgb_colourspace.whitepoint,
#                                          rgb_colourspace.matrix_XYZ_to_RGB_BT2020
#                                          )
#     return Reimg_RGB
