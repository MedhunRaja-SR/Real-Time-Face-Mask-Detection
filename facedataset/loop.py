
import os



folder_path = "C:/Users/Admin/PycharmProjects/pythonProject/mask_dataset"
#dist_path = "/home/preeth/Downloads"

c = 0
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(len(images)):
    print("the path of the image is", images[i])
    image = cv2.imread(images[i])
    c = c + 1
    create_mask(images[i])