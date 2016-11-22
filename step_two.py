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
        self.mark_colors()  # puts white wherever it thinks there is a color
        self.open()         # normal opening on those colors to blurr out any letters etc.
        self.find_contours()# find all contours
        self.contour = self.get_biggest_contour() # if there was a colorfull background
                                                  # (see img 9 with this line commented),
                                                  # you choose the biggest white contour
        #self.convex() # do I need you?


        pass

    def open(self):
        try:
            #self.altered_image = np.array(self.image[:,:,1])
            seed = (min(len(self.altered_image),len(self.altered_image[0]))**2)/10000
            print(seed)
            assert seed < 50    #na na na lag proof!



            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed)/3)
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed) / 3)
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

    def convex(self, image=None):
        if(image):
            self.altered_image = morphology.convex_hull_image(image)
        else:
            self.altered_image = morphology.convex_hull_image(self.altered_image)

    def get_biggest_contour(self):
        max_area = 0
        result = None
        for contour in self.contours:
            area = (max(contour[:,0])-min(contour[:,0])) * \
                           (max(contour[:,1]) - min(contour[:,1]))
            if max_area < area:
                max_area = area
                result = contour
        assert (result != None)
        return result

    def show(self):
        self.fig, self.plots = plt.subplots(1,2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        for contour in self.contours:
            self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
        #    self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
        self.plots[1].plot(self.contour[:, 1], self.contour[:, 0], linewidth=2, zorder=1)

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
    i=random.randint(1,14) # choose one of 14 images randomly
    #i = 11
    image = get_image(images+str(i)+".jpg", False)
    #image = get_image("pictures_small/img (7).jpg", False)
    Plamka = plamka(image)
    Plamka.process()
    Plamka.show()