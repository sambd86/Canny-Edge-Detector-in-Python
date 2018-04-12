
import cv2
import numpy as np
import PIL
import scipy.misc
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image


def histeris(retgrad):
    thresHigh=50
    thresLow=30
    init_point = stop(retgrad, thresHigh)
    # Hysteresis tracking. Since we know that significant edges are
    # continuous contours, we will exploit the same.
    # thresHigh is used to track the starting point of edges and
    # thresLow is used to track the whole edge till end of the edge.

    while (init_point != -1):
        # Image.fromarray(retgrad).show()
        # print 'next segment at',init_point
        retgrad[init_point[0], init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = nextNbd(retgrad, p0, p1, p2, thresLow)

        while (p0 != -1):
            # print p0
            p2 = p1
            p1 = p0
            retgrad[p0[0], p0[1]] = -1
            p0 = nextNbd(retgrad, p0, p1, p2, thresLow)

        init_point = stop(retgrad, thresHigh)

    # Finally, convert the image into a binary image
    x, y = np.where(retgrad == -1)
    retgrad[:, :] = 0
    retgrad[x, y] = 1.0
    return retgrad
def stop(im, thres):
    '''
        This method  finds the starting point of an edge.
    '''
    X, Y = np.where(im> thres)
    try:
        y = Y.min()
    except:
        return -1
    X = X.tolist()
    Y = Y.tolist()
    index = Y.index(y)
    x = X[index]
    return [x, y]

def nextNbd(im, p0, p1, p2, thres):
    '''
        This method is used to return the next point on the edge.
    '''
    kit = [-1, 0, 1]
    X, Y = im.shape
    for i in kit:
        for j in kit:
            if (i + j) == 0:
                continue
            x = p0[0] + i
            y = p0[1] + j

            if (x < 0) or (y < 0) or (x >= X) or (y >= Y):
                continue
            if ([x, y] == p1) or ([x, y] == p2):
                continue
            if (im[x, y] > thres):  # and (im[i,j] < 256):
                return [x, y]
    return -1

def maximum(det, phase):

  gmax = np.zeros(det.shape)

  for i in np.arange(gmax.shape[0]):

    for j in np.arange(gmax.shape[1]):

      if phase[i][j] < 0:

        phase[i][j] += 360



      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):

        # 0 degrees

        if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):

          if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:

            gmax[i][j] = det[i][j]

        # 45 degrees

        if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):

          if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:

            gmax[i][j] = det[i][j]

        # 90 degrees

        if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):

          if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:

            gmax[i][j] = det[i][j]

        # 135 degrees

        if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):

          if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:

            gmax[i][j] = det[i][j]

  return gmax

'non max ends here'
def gray_gradient(image):
    Gx, Gy = np.gradient(image)
    Gm = np.sqrt(Gx ** 2 + Gy ** 2)
    Gd = np.arctan2(Gy, Gx)
    Gd[Gd > 0.5*np.pi] -= np.pi
    Gd[Gd < -0.5*np.pi] += np.pi
    return Gm, Gd
'''gradient ends here'''
def gaussianSam(im):
    im_out=im
    height=im_out.shape[0]
    width=im_out.shape[1]
    gauss=(1.0/57)*np.array(
        [[0,1,2,1,0],
         [1,3,5,3,1],
         [2,5,9,5,2],
        [1,3,5,3,1],
        [0,1,2,1,0]])
    #sum(sum(gauss))
    for i in np.arange(2,height-2):
        for j in np.arange(2,width-2):
            sum=0
            for k in np.arange(-2,3):
                for l in np.arange(-2,3):
                    a=im.item(i+k,j+l)
                    p=gauss[2+k,2+l]
                    sum=sum+(p*a)
            b=sum
            im_out.itemset((i,j),b)

    return im_out


'''main start here'''
# im1=cv2.imread('c:/cb.png',1)
# imgRGB = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
# image00=im1
#plt.imshow(im)



im=cv2.imread('c:/cb.jpg',0)
image0=im
g=gaussianSam(im)
#print(im)
image1 = g




#gi, theta=gray_gradient(g)
#print(gi,theta)
gix=cv2.sobelx(g,(5,5),0)
giy=cv2.sobely(g,(5,5),0)
gi=np.sqrt(gix**2+giy**2)
theta=np.arctan2(giy,gix)
image2 = gi
#plt.imshow(image)


nms=maximum(gi, theta)

#plt.imshow(image)



image3 = nms
binImage=histeris(nms)
image4 = binImage
#plt.imshow(image)



#binImage=histeris(nms[0])



fig,ax=plt.subplots(2,3)
ax[0,0].imshow(image0)
#ax[0,1].imshow(image0)
ax[0,1].imshow(image1)
ax[0,2].imshow(image2)
ax[1,0].imshow(image4)
ax[1,1].imshow(image3)
plt.show()