#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# In[42]:


def get_all_image_input(directory):
    list_image = glob.glob(directory + "*.jpg")
    list_image.extend(glob.glob(directory + "*.png"))
    list_image.extend(glob.glob(directory + "*.jpeg"))
    return list_image


# In[4]:


def show_image(img):
    image=cv2.imread(img)
    plt.imshow(image)
    plt.title("Image")
    plt.show()


# In[23]:


def find_table_box(list_boxes, img_height):
    x_es=[]
    for box in list_boxes:
        x_es.append(box[0])

    results=[]
    min_height=0
    while(len(x_es)!=0):
        min_x = min(x_es)
        x_es.remove(min_x)
        min_y = img_height

        res = []
        for box in list_boxes:
            if min_x in box and box[1] < min_y and (box[1] + box[3]) > min_height:
                res = box
                min_y = box[1]
                min_height = (box[1] + box[3])

        for box in list_boxes:
            if box == res:
                results.append(res)
                list_boxes.remove(res)
    return results


# In[120]:


def get_contours(img_bin):
    # Các đường kẻ ngang + dọc
    kernel_length = np.array(img_bin).shape[1] // 80

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    img_temp1 = cv2.erode(img_bin, ver_kernel, iterations = 3)
    vertical_lines_img = cv2.dilate(img_temp1, ver_kernel, iterations=3)

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)


    # matrix 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # img_final_bin = vertical_lines_img + horizontal_lines_img
    img_final_bin = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    img_final_bin = 255 - img_final_bin
    img_final_bin = cv2.erode(img_final_bin, kernel, iterations=2)

    (thresh, img_final) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Tìm các contour của ảnh
    im, contours, hierarchy = cv2.findContours(img_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return (im, contours, hierarchy, img_final)


# In[130]:


def main(list_image):
    for image in list_image:
        print("\nProcessing", image, "---------")

        # Tạo thư mục output
        basename = os.path.basename(image)
        filename = os.path.splitext(basename)[0]
        dir = './output/' + filename
        if not os.path.exists(dir):
            os.mkdir(dir)

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape[:2]
        (thresh, img_bin) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        img_bin = 255 - img_bin

        (im, contours, hierarchy, img_final) = get_contours(img_bin)

        contours.reverse()
        boxes=[]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x != 0 and y !=0 and w != width and h != height:
#                 print(x,"\t", y,"\t", x+w,"\t", y+h)
#                 cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 0, 0), 1)
                boxes.append([x,y,w,h])

        # ----------------
        table = find_table_box(boxes, height)
        for t in table:
            x,y,w,h = t[:]
            img_table = img[y:y+h, x:x+w]

            (thresh, img_bin) = cv2.threshold(img_table, 200, 255, cv2.THRESH_BINARY)
            img_bin = 255 - img_bin

            (_im, _contours, _hierarchy, _img) = get_contours(img_bin)
            contours.reverse()
            for contour in _contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 1.5*h:
                    cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
                    print("Locate:",x, y, w, h)
            cv2.imwrite("./output/"+filename+"/_img"+str(table.index(t))+".png", _img)


# In[131]:


input = get_all_image_input('./input/')
print(input)


# In[132]:


main(input)


# In[ ]:





# In[ ]:


def test():
    cv2.imwrite("./output/"+filename+"/img_final_bin.png",img_final_bin)
    print("Saved img_final_bin.png")

    new_img=cv2.imread("./output/"+filename+"/img_final_bin.png")
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(new_img, (x,y), (x+w,y+h),(0,255,0),1)
    cv2.imwrite("./output/"+filename+"/new_img.png",new_img)

