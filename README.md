# color_enhancement

This code is to enhance color without color shift and output a cube file for the real time application.
You can apply this cube file to an image or a video.

Due to the differences between the CIE 1931 and CIE LUV color spaces, the results differ when mapping linear color enhancements. It can be observed that color enhancement in the CIE LUV color space results in more uniform improvements.

##EDSR Upscaling withtout Color Enhancement Algorithm
![EDSR_Upscaling_without_ColorEnhancement](https://github.com/suk1998/color_enhancement/blob/main/Gettyimages-1435476284_EDSR_jpeg70.jpg)


##EDSR Upscaling with CIE 1931 Color Enhancement Algorithm

![CIE1931_applied_Gettyimages-1435476284_EDSRl_jpeg70](https://github.com/user-attachments/assets/c296438c-14fc-4823-ac99-266a61d7100d)


##EDSR Upscaling with CIE LUV Color Enhancement Algorithm


![CIELUV_LUT1.4_applied_Gettyimages-1435476284_EDSR_jpeg70.jpg](https://github.com/suk1998/color_enhancement/blob/main/CIELUV_LUT1.4_applied_Gettyimages-1435476284_EDSR_jpeg70.jpg)






reference: https://hs-paik.netlify.app/cieluv_color


