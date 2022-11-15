import os
from PIL import Image 

#folder_images = "C:/Users/Admin/AppData/Local/Programs/Python/Python310/facemask/dataset/with_mask"
folder_images = r"C:\Users\Admin\AppData\Local\Programs\Python\Python310\facemask\dataset\with_mask"
size_images = dict()
print("With Mask...")

for dirpath, _, filenames in os.walk(folder_images):
    for path_image in filenames:
        image = os.path.abspath(os.path.join(dirpath, path_image))
        with Image.open(image) as img:
            width, height = img.size
            size_images[path_image] = {'width':width,'height': height}
            #print(img.shape)
			
print(size_images)
           
