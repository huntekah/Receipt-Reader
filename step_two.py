import paragon_proscessing
from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import random

SHOW_CONTOURS = True

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
        #self.print_values_on_contour(self.contour)
        # just to learn kwargs! :D
        #self.show()
        self.altered_image = morphology.convex_hull_object(self.altered_image, 8)
        #self.show()
        self.image = self.image_X_mask()
        self.show()
        self.image, self.altered_image = self.trim_to_mask(source=self.image, mask=self.convex(self.altered_image))
        #self.gray_image = rgb2gray(self.image)
        self.show()
        self.altered_image = self.erase_colors(0.0)
        self.open(5)
        self.find_contours()

        #self.convex() # do I need you?


        pass

    def trim_to_mask(self, **kwargs):
        options = {
            'source'    : self.image,
            'mask'      : self.altered_image
        }
        options.update(kwargs)

        image = options['source']
        altered_image = options['mask']

        region = self.choose_mask(altered_image)
        print(region.bbox)
        altered_image = region.filled_image
        image = self.trim(image, region.bbox)
        return image, altered_image

    def choose_mask(self,image=None):
        if image is None:    #default argument
            image = self.altered_image

        self.label_img = measure.label(image,neighbors=8)
        self.regions = measure.regionprops(self.label_img)
        best_region = self.regions[0]
        for property in self.regions:
            if best_region.area < property.area:
                best_region = property
        return best_region

    def print_values_on_contour(self, contour):
        for pixel in contour:
            print(self.altered_image[pixel[0]][pixel[1]])

    def trim(self, source, bbox=None, contour=None):
        '''returns the image in the smallest box containing contour or by bbox coordinates'''
        if bbox is None and contour is None:
            bbox = (0, 0, len(source), len(source[0]))
        if not (contour is None):
            return source[min(contour[:, 0]): max(contour[:, 0]), min(contour[:, 1]): max(contour[:, 1])]
        else:
            return source[bbox[0]: bbox[2], bbox[1]: bbox[3]]

    #should wrap in decorator!
    def open(self, alpha=2.5):
        try:
            #self.altered_image = np.array(self.image[:,:,1])
            seed = (min(len(self.altered_image),len(self.altered_image[0]))**2)/10000
            print(seed)
            assert seed < 50    #na na na lag proof!



            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / alpha))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / alpha))
            #self.altered_image = filters.gaussian(self.altered_image,sigma = seed)
        except AssertionError as e:
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(10))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(10))
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def close(self, alpha=2.5):
        try:
            # self.altered_image = np.array(self.image[:,:,1])
            seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
            print(seed)
            assert seed < 50  # na na na lag proof!

            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / alpha))
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / alpha))
            # self.altered_image = filters.gaussian(self.altered_image,sigma = seed)
        except AssertionError as e:
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed/alpha**2))
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed/alpha**2))
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def mark_colors(self,threshold=0.25):
        for i, row in enumerate(self.image):
            for j, rgb in enumerate(row):
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                grey = (int(r)+int(g)+int(b))/3
                distance = abs(r-grey) + abs(g-grey) + abs(b-grey)
                if distance / 255 > threshold:    # image is 0..255 although altered is 0..1
                    self.altered_image[i][j] = 1
                else :
                    self.altered_image[i][j] = 0

    def find_contours(self):
        threshold = 0.7 #powinno zależeć od wartości kolorów w obrazku, a nie być ustalane na sztywno
        connected = 'low'
        self.contours = measure.find_contours(self.altered_image, level=threshold , fully_connected=connected)



    def convex(self, image=None):
        if image is None:
            return morphology.convex_hull_image(image)
        else:
            return morphology.convex_hull_image(self.altered_image)

    def get_biggest_contour(self):
        max_area = 0
        result = None
        for contour in self.contours:
            area = (max(contour[:,0])-min(contour[:,0])) * \
                           (max(contour[:,1]) - min(contour[:,1]))
            if max_area < area:
                max_area = area
                result = contour
        assert not (result is None)

        return result

    def image_X_mask(self, **kwargs):
        options = {
            'source'    : self.image,
            'mask'      : self.altered_image,
            'background': 1
        }
        options.update(kwargs)

        image = options['source']
        altered_image = options['mask']
        background = options['background']

        for i, row in enumerate(image):
            for j, pixel in enumerate(row):
                if altered_image[i][j] == 0:
                    image[i][j] = [255 * background,
                                   255 * background,
                                   255 * background]
                else:
                    image[i][j] = [pixel[0] * altered_image[i][j],\
                               pixel[1] * altered_image[i][j],\
                               pixel[2] * altered_image[i][j]]
        return image

    def erase_colors(self,threshold=0.4, **kwargs):
        options = {
            'image': self.image,
        }
        options.update(kwargs)

        image = options['image']

        img = np.ones((len(image), len(image[0])))
        for i, row in enumerate(image):
            for j, rgb in enumerate(row):
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                grey = (int(r)+int(g)+int(b))/3
                distance = abs(r-grey) + abs(g-grey) + abs(b-grey)
                if distance / 255 >= threshold:    # image is 0..255 although altered is 0..1
                    grey = max(rgb)
                    img[i][j] = grey/255
        return img

    def show(self):
        self.fig, self.plots = plt.subplots(1,2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        if SHOW_CONTOURS:
            for contour in self.contours:
                self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
                self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
          #  self.plots[1].plot(self.contour[:, 1], self.contour[:, 0], linewidth=2, zorder=1)

        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal",fontsize=10)
        self.plots[1].set_title("altered",fontsize=10)

        plt.show()

def get_image(path, asgrey=True, _flatten=False):
    print("loading image "+path)
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


if __name__ == "__main__":
    images = "pictures_small/plamka"

    #for i in range(1,1):
    i=random.randint(1,14) # choose one of 14 images randomly
    i = 1 #to chooose specyfic image
    image = get_image(images+str(i)+".jpg", False)
    #image = get_image("pictures_small/img (7).jpg", False)
    Plamka = plamka(image)
    Plamka.process()
    Plamka.show()