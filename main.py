import copy
import math

import cv2
import numpy as np


def multiply(a, b):
    if (len(a[0]) != len(b)):
        print("The two matricies are not multiplicable")
        return
    result = [[0]*len(b[0]) for i in range(len(a))]
    
    for i in range(0, len(a)):
        for j in range(0, len(b[0])):
            sum = 0

            for k in range(0, len(a[0])):
                sum += a[i][k] * b[k][j]

                #print(i," ", j," ",k," ")
            result[i][j] = sum






    return result





def makeBuffer(img):
    numRows = img.shape[0]
    numCols = img.shape[1]

    scalingFactor = 1.3
    maxDim = max(numRows, numCols)
    n = scalingFactor * maxDim
    n = round(n)

    emptyIm = np.zeros((n, n, 3), np.float32)

    centerEmp = (round(n / 2), round(n / 2))
    centerOrg = (round(numRows / 2), round(numCols / 2))
    displaceX = centerEmp[1] - centerOrg[1]
    displaceY = centerEmp[0] - centerOrg[0]

    for i in range(numRows):
        for j in range(numCols):
            # [i][j][0]: blue
            # [i][j][1]: green
            # [i][j][2]: red
            emptyIm[i + displaceY][j + displaceX][0] = img[i][j][0]
            emptyIm[i + displaceY][j + displaceX][1] = img[i][j][1]
            emptyIm[i + displaceY][j + displaceX][2] = img[i][j][2]
    return emptyIm


def rotate(angle,img,r):
    rotationArr = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    tempImg = np.zeros((img.shape[0],img.shape[1],3),np.float32)




    centerX = round(img.shape[1] / 2)
    centerY = round(img.shape[0] / 2)




    itransposed = 0
    jtransposed = 0


    rerr = r

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):




            itransposed = i - centerY
            jtransposed = j - centerX

            arr = [[jtransposed], [itransposed]]
            res = multiply(rotationArr,arr)

            irot = res[1][0]
            jrot = res[0][0]

            ifin = irot + centerY
            jfin = jrot + centerX

            if((round(ifin) < img.shape[0] and round(jfin) < img.shape[1]) and (round(ifin) > 0 and round(jfin) > 0)):


                tempImg[i][j][0] = img[int(round(ifin))][int(round(jfin))][0]
                tempImg[i][j][1] = img[int(round(ifin))][int(round(jfin))][1]
                tempImg[i][j][2] = img[int(round(ifin))][int(round(jfin))][2]








            rerr += math.sqrt(math.pow(ifin - round(ifin),2) + (math.pow(jfin - round(jfin),2)))
    print(rerr)




    cv2.imshow("test",tempImg/255.0)
    cv2.waitKey()
    return tempImg



def rotate2pi(img,steps):
    stepSize = (2*math.pi)/steps
    r = 0
    orgim = copy.deepcopy(img)
    for i in range(steps):

        img = rotate(stepSize,img,r)
    diffB = 0
    diffG = 0
    diffR = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            diffB += abs(img[i][j][0] - orgim[i][j][0])
            diffR += abs(img[i][j][1] - orgim[i][j][1])
            diffG += abs(img[i][j][2] - orgim[i][j][2])
    print(diffB)
    print(diffG)
    print(diffR)























def main():
    img = cv2.imread("tigers.jpg")












    image = makeBuffer(img)

    rotate2pi(image,4)








if __name__ == "__main__":
    main()