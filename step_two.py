import paragon_proscessing
from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import random

class plamka:
    def __init__(self,image):
        self.image = np.array(image)
        self.gray_image = rgb2gray(image)
        self.altered_image = rgb2gray(image)

    def process(self):
        self.mark_colors()
        self.find_contours()
        self.open()
        pass

    def open(self):
        try:
            #self.altered_image = np.array(self.image[:,:,1])
            seed = (min(len(self.altered_image),len(self.altered_image[0]))**2)/10000
            print(seed)
            assert seed < 50    #na na na lag proof!



            #self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed))
            #self.altered_image = filters.gaussian(self.altered_image,sigma = seed)
        except AssertionError as e:
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def mark_colors(self):

        for i, row in enumerate(self.image):
            for j, rgb in enumerate(row):
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                grey = (int(r)+int(g)+int(b))/3
                distance = abs(r-grey) + abs(g-grey) + abs(b-grey)
                if distance / 255 > 0.25:    # image is 0..255 although altered is 0..1
                    self.altered_image[i][j] = 1
                else :
                    self.altered_image[i][j] = 0

    def find_contours(self):
        threshold = 0.7 #powinno zależeć od wartości kolorów w obrazku, a nie być ustalane na sztywno
        connected = 'low'
        self.contours = measure.find_contours(self.altered_image, level=threshold , fully_connected=connected)


    def show(self):
        self.fig, self.plots = plt.subplots(1,2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        for contour in self.contours:
            self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
        #    self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)

        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal",fontsize=10)
        self.plots[1].set_title("altered",fontsize=10)

        plt.show()

def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


if __name__ == "__main__":
    images = "pictures_small/plamka"

    #for i in range(1,1):
    i=random.randint(1,14)

    image = get_image(images+str(i)+".jpg", False)
    Plamka = plamka(image)
    Plamka.process()
    Plamka.show()