"""
BT.2020 UHD 4K 8K 를 위한 색 매핑 코드
색영역: CIELuv
2023/05/09
백희숙

"""

import numpy as np
import matplotlib.pyplot as plt
import colour
from colour.models import RGB_COLOURSPACE_BT2020

# Library for creating 3D LUT
from colour.io import write_LUT_IridasCube
from colour import LUT3D

ILLUMINANT_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
rgb_colourspace = colour.models.RGB_COLOURSPACE_BT2020.chromatically_adapt(ILLUMINANT_D65)

def gamut_mapping_luv(img_luv, enhancement_ratio=1.0):
    white_point = np.array([100.0, 0.0, 0.0])

    for i in range(img_luv.shape[0]):
        for j in range(img_luv.shape[1]):
            direction = img_luv[i,j,1:] - white_point[1:]
            img_luv[i,j,1:] = white_point[1:] + direction * enhancement_ratio
    
    return img_luv

file_path = './images/color-test-file.jpg'
img = colour.read_image(file_path)

# UHDRGB to XYZ
#illuminant = np.array([0.34570, 0.35850])
img_xyz = colour.RGB_to_XYZ(img, rgb_colourspace.whitepoint,
                                   rgb_colourspace.whitepoint,
                                   rgb_colourspace.matrix_RGB_to_XYZ)

#XYZ to CIELUV
img_luv = colour.XYZ_to_Luv(img_xyz)

#Gamut Mapping in CIELUV
img_luv_mapped = gamut_mapping_luv(np.copy(img_luv))

#CIELUV to XYZ
img_xyz_mapped = colour.Luv_to_XYZ(img_luv_mapped)

# XYZ to UHDRGB
img_rgb_mapped = colour.XYZ_to_RGB(img_xyz_mapped, rgb_colourspace.whitepoint,
                                         rgb_colourspace.whitepoint,
                                         rgb_colourspace.matrix_XYZ_to_RGB)
img_rgb_mapped = np.clip(img_rgb_mapped, 0,1)

img_rgb_mapped = np.asarray(img_rgb_mapped*255, dtype=np.uint8)


colour.write_image(img_rgb_mapped,'./images/reproduced_CIELUV_BT2020_color-test-file_1.4_2.jpg', bit_depth='uint8')

############################ Creating 3D LUT #####################################
print()
print('Start to creat a 3D LUT !!! *************************************')

white_point = np.array([100.0, 0.0, 0.0])

lut_RGB = LUT3D.linear_table(64)

lut_xyz = colour.RGB_to_XYZ(lut_RGB,rgb_colourspace.whitepoint,
                                   rgb_colourspace.whitepoint,
                                   rgb_colourspace.matrix_RGB_to_XYZ)

lut_luv = colour.XYZ_to_Luv(lut_xyz)

lut_luv_L = lut_luv[...,:1]

lut_uv = lut_luv[...,1:]

for i in range(len(lut_uv)):
    for j in range(len(lut_uv[i])):
        for k in range(len(lut_uv[i][j])):
            direction = lut_uv[i][j][k] - white_point[1:]
            lut_uv[i][j][k] = lut_uv[i][j][k] + white_point[1:] + direction * 1.0

lut_Luv = np.append(lut_luv_L, lut_uv, axis=3)

lut_xyz_mapped = colour.Luv_to_XYZ(lut_Luv)

lut_rgb_mapped = colour.XYZ_to_RGB(lut_xyz_mapped, rgb_colourspace.whitepoint,
                                         rgb_colourspace.whitepoint,
                                         rgb_colourspace.matrix_XYZ_to_RGB)

lut_rgb_mapped = np.clip(lut_rgb_mapped, 0, 1)

LUT = LUT3D(
    lut_rgb_mapped,
    'CIELUV_LUT',
#     domain,

    comments=['CIELUV BT2020_to_BT2020', 'grid=127'])

write_LUT_IridasCube(LUT, './LUTs/CIELUV_BT2020_LinearExpansion_64.cube') 

# Applying LUT
lut_applied_img = LUT.apply(img)
plt.title('LUT_applied_Image')
plt.imshow(lut_applied_img)
plt.axis('off')
plt.show()

colour.write_image(lut_applied_img,'./result/LUT_applied_CIELUV_BT2020_color-test-file_1.2.jpg', bit_depth='uint8')

print("End")