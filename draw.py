import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw(img_array, filename):
        mpimg.imsave(filename, np.concatenate( [img_array.reshape([-1,1])] *8, axis=1).reshape([267,248]) )

def main():
    if len(sys.argv)>=1:
        file_name = sys.argv[1]
    if len(sys.argv)>=2:
        destpath = sys.argv[2]
    f = np.loadtxt(file_name)
    #f = f.reshape( [ 366, 30, 1 ] )
    #img = np.concatenate( [ f, f, f], axis=2 )
    img = np.concatenate( [f]*12, axis=1 )
    img = img.reshape([267,31*8])
    if len(sys.argv)>=2:
        mpimg.imsave(destpath , img)
    else:
         mpimg.imsave("result_img/" + file_name[:-3] + "png", img)

if __name__ == "__main__":
    main()
