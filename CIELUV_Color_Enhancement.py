import numpy as np
import matplotlib.pyplot as plt
import colour

def gamut_mapping_luv(img_luv, enhancement_ratio=1.4):
    white_point = np.array([100.0, 0.0, 0.0])

    for i in range(img_luv.shape[0]):
        for j in range(img_luv.shape[1]):
            direction = img_luv[i,j,1:] - white_point[1:]
            img_luv[i,j,1:] = white_point[1:] + direction * enhancement_ratio
    
    return img_luv

file_path = './images/IA0008358001_SR.mp4_20230503_111346.png'
img = colour.read_image(file_path)
print("Original Image's RGB Values =======================================>")
print(img)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()
#sRGB to XYZ
img_xyz = colour.sRGB_to_XYZ(img)
print("Image XYZ Values =======================================>")
print(img_xyz)
#XYZ to CIELUV
img_luv = colour.XYZ_to_Luv(img_xyz)
print("Image CIELUV Values =======================================>")
print(img_luv)
#Gamut Mapping in CIELUV
img_luv_mapped = gamut_mapping_luv(np.copy(img_luv))
print("Image CIELUV Mapped Values =======================================>")
print(img_luv_mapped)

#CIELUV to XYZ
img_xyz_mapped = colour.Luv_to_XYZ(img_luv_mapped)
print("Image XYZ Mapped Values =======================================>")
print(img_xyz_mapped)

#XYZ to sRGB
img_rgb_mapped = colour.XYZ_to_sRGB(img_xyz_mapped)
img_rgb_mapped = np.clip(img_rgb_mapped, 0,1)
print("Image RGB Mapped Values =======================================>")
print(img_rgb_mapped)

img_rgb_mapped = np.asarray(img_rgb_mapped*255, dtype=np.uint8)
print("uint8_Image RGB Mapped Values =======================================>")
print(img_rgb_mapped)
plt.imshow(img_rgb_mapped)
plt.title('Enhanced Image')
plt.axis('off')
plt.show()

colour.write_image(img_rgb_mapped,'./images/reproduced_CIELUV_IA0008358001_SR.mp4_20230503_111346.png', bit_depth='uint8')