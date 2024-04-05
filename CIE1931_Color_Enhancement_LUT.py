"""
### Creating a 3D Color Lookup Table
AdobeRGB 는 reference destination gamut 으로 사용
BT.709 to BT.709
3 NP to change the intensity of color
Saturation Mapping Function 은 norm.pdf 대신 수식으로 사용
DS = Different Saturation Mapping function
"""
import sys
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import colour
from colour.plotting import *

from RGB_to_xyY import sRGB_to_xyY, xyY_to_sRGB

# Library for creating 3D LUT
from colour.io import write_LUT_IridasCube
from colour import LUT3D

sys.path.append('colour')
colour.plotting.colour_style()

import time
start = time.time()

# Setting hyper parameters
file_name = 'MacbethCC-sRGB.jpg'

MaxStep = 0.08
sig = 0.05
MeanPercent = 0.5

# Setting Source gamut 
Sx = np.array([0.6400, 0.3000, 0.1500])
Sy = np.array([0.3300, 0.6000, 0.0600])
# Setting Destination gamut
Adx = np.array([0.6400, 0.2100, 0.1500])
Ady = np.array([0.3300, 0.7100, 0.0600])
Adxy = np.array([[0.64, 0.33],[0.21,0.71],[0.15,0.06]])
Adxy_R = np.array([0.6400, 0.3300])
Adxy_G = np.array([0.2100, 0.7100])
Adxy_B = np.array([0.1500, 0.0600])

# White Point 설정 D65(6000k)
D65x = np.array([0.3127])
D65y = np.array([0.3290])
Origin = np.block([0.3127, 0.3290])

# Middle Point 설정
# BT.709 gamut 에서 각각 R에서 G 까지의 중간 지점 포인트를 정한다.
midPnt_RG_x = np.array([.47])
midPnt_RG_y = np.array([.4649])
midPnt_GB_x = np.array([.225])
midPnt_GB_y = np.array([.329])
midPnt_BR_x = np.array([.395])
midPnt_BR_y = np.array([.195])

# Loading Images
img = colour.read_image(file_name)

# 예외 처리
if img is None:
    print('Image Loading Failed!!')
    sys.exit()

print(img.shape, img.size, img.ndim, img.dtype)
height, width, channel = img.shape
print(height)

# BT.709 sRGB to xyY 변환
img_xyY = sRGB_to_xyY(img)
print("ImgxyY=================================================")
print(img_xyY)
print(img_xyY.shape, img_xyY.size, img_xyY.ndim, img_xyY.dtype)

# Seperate Y value
imgxyY_Y = img_xyY[:, :, 2]
imgxyY_Y = imgxyY_Y.reshape(height,width, 1)

print("ImgxyY_Y=================================================")
print(imgxyY_Y)

# xyY to xy 변환
img_xy = colour.xyY_to_xy(img_xyY)
print('imgxy.shape:', img_xy.shape)
print("Imgxy=================================================")
print(img_xy)

############################### CIE xy Color Space #################################

# BT.709 gamut 을 midPnt 3개의 영역으로 나눔
R_polygon = [(0.6400, 0.3300),(0.47,0.4649),(0.3127, 0.3290),(0.395, 0.195)]
R_poly = Polygon(R_polygon)
G_polygon = [(0.3000, 0.6000),(0.225, 0.329),(0.3127, 0.3290),(0.47,0.4649)]
G_poly =Polygon(G_polygon)
B_polygon = [(0.1500, 0.0600),(0.395, 0.195),(0.3127, 0.3290),(0.225, 0.329)]
B_poly = Polygon(B_polygon)

In_R_Poly = 0
In_G_Poly = 1
In_B_Poly = 2
In_X_Poly = 3 # Not in R,G,B poly, out of gamut 

# 3개의 영역에 포함되는 칼라 포인트들을 각각 In_R_poly, In_G_Poly, In_B_Poly로 나누는 함수
def RGB_region(imgxy_value) :
    if R_poly.contains(Point(tuple(imgxy_value))) :
        return In_R_Poly
    elif G_poly.contains(Point(tuple(imgxy_value))) :
        return In_G_Poly
    elif B_poly.contains(Point(tuple(imgxy_value))) :
        return In_B_Poly
    else :
        return In_X_Poly
    
# Extract exterior coordinates 시각화
R_x, R_y = R_poly.exterior.coords.xy
G_x, G_y = G_poly.exterior.coords.xy
B_x, B_y = B_poly.exterior.coords.xy

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.plot(R_x, R_y, color='red', linewidth=2, label='R')
ax.plot(G_x, G_y, color='green', linewidth=2, label='G')
ax.plot(B_x, B_y, color='blue', linewidth=2, label='B')

scatter_x=img_xy[:,:,0].reshape(-1)
scatter_y=img_xy[:,:,1].reshape(-1)

plt.scatter(scatter_x,scatter_y, c='black')
plt.show()

########################## Saturation Mapping Function ###################################
R_Maxdis = np.linalg.norm((Adx[0], Ady[0])-Origin)
G_Maxdis = np.linalg.norm((Adx[1], Ady[1])-Origin)
B_Maxdis = np.linalg.norm((Adx[2], Ady[2])-Origin)

print("R,G,B Maxdis:",R_Maxdis,G_Maxdis,B_Maxdis)

R_disMean = R_Maxdis*MeanPercent
G_disMean = G_Maxdis*MeanPercent
B_disMean = B_Maxdis*MeanPercent

# Red Distance Distribution
xn1 = np.arange(0, R_Maxdis, 0.001)
yn1 = np.exp(-0.5 * ((xn1 - R_disMean) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
Rdegree_ratio = MaxStep / np.max(yn1)
yn1 *= Rdegree_ratio

# Green Distance Distribution
xn2 = np.arange(0, G_Maxdis, 0.001)
yn2 = np.exp(-0.5 *((xn2 - G_disMean)/ sig) ** 2) / (sig * np.sqrt(2 * np.pi))
Gdegree_ratio = MaxStep / np.max(yn2)
yn2 *= Gdegree_ratio

# Blue Distance Distribution
xn3 = np.arange(0, B_Maxdis, 0.001)
yn3 = np.exp(-0.5 *((xn3 - B_disMean)/ sig) ** 2) / (sig * np.sqrt(2 * np.pi))
Bdegree_ratio = MaxStep/ np.max(yn3)
yn3 *= Bdegree_ratio


plt.plot(xn1, yn1, 'r-')
plt.plot(xn2, yn2, 'g-')
plt.plot(xn3, yn3, 'b-')

plt.show()

################### Linear Expansion ################################
# 매핑된 이미지 xy 데이터들이 array 로 Reimg_xy = []에 저장됨
Reimg_xy= []
Reimg_xy_RGB=[]
# r_sq 초기화
r_sq = 0

#R,G,B 영역 count
rcount=0
gcount=0
bcount=0
xcount=0

# distance 와 direction 설정
for i in tqdm(range(len(img_xy))):
    Reimg_xy.append([])
    Reimg_xy_RGB.append([])
    for j in range(len(img_xy[i])):                  
        dist = np.linalg.norm(img_xy[i][j]-Origin)

        #220719 dist(pnt와 white point의 거리)가 0일 때 imgxy 그대로 처리
        if abs(dist) <= sys.float_info.epsilon :
            Reimg_xy[i]. append(img_xy[i][j])
            continue


        dir = (img_xy[i][j]-Origin)/dist
        
        if RGB_region(img_xy[i][j]) == In_R_Poly:
            disMean = R_disMean
            Max_dis = R_Maxdis
            degree_ratio = Rdegree_ratio
            
            rcount+=1


        elif RGB_region(img_xy[i][j]) == In_G_Poly:
            disMean = G_disMean
            Max_dis = G_Maxdis
            degree_ratio = Gdegree_ratio
            
            gcount+=1



        elif RGB_region(img_xy[i][j]) == In_B_Poly:
            disMean = B_disMean
            Max_dis = B_Maxdis
            degree_ratio = Bdegree_ratio
            
            bcount+=1

        else:
            
            disMean = 0
            Max_dis = 0
            degree_ratio = 0
            xcount+=1

        r_sq = norm.pdf(dist, disMean, sig)*degree_ratio
        Reimg_xy[i].append(img_xy[i][j] + r_sq * dir)

print("Reimg_xy=================================================")
Reimg_xy = np.array(Reimg_xy)
print(Reimg_xy)

print("Rcount:",rcount,"Bcount:",bcount,"Gcount:",gcount,"Xcount:",xcount,"sum:",rcount+bcount+gcount+xcount)
print("#pixel:",height*width)
print()
plt.plot(D65x,D65y,'+', markersize=15, color='red')
plt.plot(midPnt_RG_x, midPnt_RG_y,'o', markersize=5, color='red')
plt.plot(midPnt_GB_x, midPnt_GB_y,'o', markersize=5, color='green')
plt.plot(midPnt_BR_x, midPnt_BR_y,'o', markersize=5, color='blue')

# BT.709 gamut을 convexhull 로 연결
sRGB = np.array([list(element) for element in zip(Sx,Sy)])
hull = ConvexHull(sRGB)
plt.scatter(Sx,Sy)
plt.axis('equal')
plt.grid(True)

for simplex in hull.simplices:
    plt.plot(sRGB[simplex, 0], sRGB[simplex, 1], 'k--')

# AdobeRGB gamut 을 ConvexHull 로 연결
AdobeRGB = np.array([list(element) for element in zip(Adx, Ady)])
hull = ConvexHull(AdobeRGB)
plt.scatter(Adx,Ady)
plt.axis('equal')
plt.grid(True)

for simplex in hull.simplices:
    plt.plot(AdobeRGB[simplex, 0], AdobeRGB[simplex, 1], 'k-')

scatter_rex=Reimg_xy[:,:,0].reshape(-1)
scatter_rey=Reimg_xy[:,:,1].reshape(-1)
plt.scatter(scatter_rex,scatter_rey, c='yellow')
plt.axis('equal')
plt.grid(True)
plt.show()

# mapped xy + imgxyY_Y

Reimg_xyY = np.append(Reimg_xy, imgxyY_Y, axis=2)
print("Reimg_xyY=================================================")
print(Reimg_xyY)

# Reimg_RGB 를 sRGB 로 변환
Reimg_RGB = xyY_to_sRGB(Reimg_xyY)

print("Reimg_RGB==================================================")
print(Reimg_RGB)

Reimg_RGB = np.clip(Reimg_RGB,0,1)

print("Reimg_RGB MAX 1 MIN 0 ==================================================")
print(Reimg_RGB)

# RGB (0 ~ 255)으로 변환
Reimg_RGB = np.asarray(Reimg_RGB * 255, dtype=np.uint8)
print('Shape:',Reimg_RGB.shape)
print('Dtype:',Reimg_RGB.dtype)
print("Reimg_RGB_uint8==================================================")
print(Reimg_RGB)
plt.imshow(Reimg_RGB)
plt.axis('off')
plt.show()

# 변환된 이미지 저장
reproduced_img = colour.write_image(Reimg_RGB, 'reproduced_BT709_diff_saturation_result.png', bit_depth='uint8')

############################ Creating 3D LUT #####################################
print('Start to creat a 3D LUT !!! *************************************')

lut_RGB = LUT3D.linear_table(127)

r_sq = 0
rcount=0
gcount=0
bcount=0
xcount=0

#xyY 변환
lut_xyY = sRGB_to_xyY(lut_RGB)
lut_xyY_Y = np.delete(lut_xyY, 0, axis=3)
lut_xyY_Y = np.delete(lut_xyY_Y, 0, axis=3)

lut_xy = colour.xyY_to_xy(lut_xyY)

print("=================================lut_xy===================================")
print(lut_xy)


# distance 와 direction 설정
print("processing LUT... ")
for i in tqdm(range(len(lut_xy))):
    for j in range(len(lut_xy[i])):
        for k in range(len(lut_xy[i][j])):
            dist = np.linalg.norm(lut_xy[i][j][k]-Origin)

            #dist(pnt와 white point의 거리)가 0일 때 LUT 그대로 처리
            if abs(dist) <= sys.float_info.epsilon :
                continue
            dir = (lut_xy[i][j][k]-Origin)/dist
            
            if RGB_region(lut_xy[i][j][k]) == In_R_Poly:
                disMean = R_disMean
                Max_dis = R_Maxdis
                degree_ratio = Rdegree_ratio
                
                rcount+=1


            elif RGB_region(lut_xy[i][j][k]) == In_G_Poly:
                disMean = G_disMean
                Max_dis = G_Maxdis
                degree_ratio = Gdegree_ratio
                
                gcount+=1
            
            elif RGB_region(lut_xy[i][j][k]) == In_B_Poly:
                disMean = B_disMean
                Max_dis = B_Maxdis
                degree_ratio = Bdegree_ratio
                
                bcount+=1

            else:
                
                disMean = 0
                Max_dis = 0
                degree_ratio = 0
                xcount+=1

            r_sq = norm.pdf(dist, disMean, sig)*degree_ratio
            lut_xy[i][j][k] = lut_xy[i][j][k] + r_sq * dir

        


print("Rcount:",rcount,"Gcount:",gcount,"Bcount:",bcount,"Xcount:",xcount,"Sum:",rcount+gcount+bcount+xcount)                  

lut_xyY = np.append(lut_xy, lut_xyY_Y, axis=3)
print("****** lut_xyY *******")
print(lut_xyY)
lut_RGB = xyY_to_sRGB(lut_xyY)
print()
lut_RGB = np.clip(lut_RGB,0,1)
print(lut_RGB)
print(lut_RGB.shape)



LUT = LUT3D(
    lut_RGB,
    'My LUT',
#     domain,

    comments=['BT709_to_BT709.', 'grid=127'])




write_LUT_IridasCube(LUT, 'BT_709_BT709_LinearExpansion.cube')   
end = time.time()
print(f"{end-start:.5f}sec")
plt.title('Source Image')
plt.imshow(img)
plt.axis('off')
plt.show()


# apply lut 
lut_arr = LUT.apply(img)
plt.title('LUT_applied_Image')
plt.imshow(lut_arr)
plt.axis('off')
plt.show()

colour.write_image(lut_arr,'LUT_applied_BT709_result.png', bit_depth='uint8')

print("End")
