# get the image from "https://cdn.pixabay.com/photo/2017/03/27/16/50/beach-2179624_960_720.jpg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.image import imread

def image_compre():
    # read image in grayscale
    img = cv2.imread('beach-2179624_960_720.jpg', 0)

    # obtain svd
    U, S, V = np.linalg.svd(img)

    # inspect shapes of the matrices
    print(U.shape, S.shape, V.shape)

    # plot images with different number of components
    comps = [638, 500, 400, 300, 200, 100]

    fig = plt.figure(figsize = (16, 8))
    for i in range(6):
      low_rank = U[:, :comps[i]] @ np.diag(S[:comps[i]]) @ V[:comps[i], :]
      if(i  == 0):
         plt.subplot(2, 3, i+1), plt.imshow(low_rank, cmap = 'gray'), plt.axis('off'), plt.title("Original Image with n_components =" + str(comps[i]))
      else:
         plt.subplot(2, 3, i+1), plt.imshow(low_rank, cmap = 'gray'), plt.axis('off'), plt.title("n_components =" + str(comps[i]))
    fig.show()


    A=imread('dog.jpg');


    plt.rcParams['figure.figsize'] = [16, 8]


    X = np.mean(A, -1); # Convert RGB to grayscale

    img = plt.imshow(X)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()

    U, S, VT = np.linalg.svd(X,full_matrices=False)
    S = np.diag(S)

    j = 0
    for r in (5, 20, 100):
        # Construct approximate image
        Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:] #rank = r
        plt.figure(j+1)
        j += 1
        img = plt.imshow(Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))
        plt.show()

    plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title('Singular Values')
    plt.show()

    plt.figure(2)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Singular Values: Cumulative Sum')
    plt.show()


if __name__ == '__main__':
    image_compre()
