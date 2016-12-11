import paragon_proscessing
from skimage import data, io, filters, feature, morphology, measure, exposure, color
# from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

SHOW_CONTOURS = False
# 0..9 + 10 as nothingness
MOMENTS_HU = [
    [5.67387500e-01, 9.21207734e-02, 1.31548831e-04, 4.16906116e-05, 1.92353804e-09, 3.27366905e-06, -2.41503441e-09],
    [7.95980240e-01, 5.61109524e-01, 8.55233907e-02, 5.61481759e-02, 3.87583725e-03, 4.14445378e-02, -3.41613871e-04],
    [6.68827215e-01, 2.35039955e-01, 1.85919959e-03, 2.75797883e-03, 6.19918694e-06, 1.21394435e-03, -7.57068092e-07],
    [6.06211220e-01, 1.68984849e-01, 8.93090218e-03, 2.26979587e-03, -7.51558195e-06, -8.35967378e-04, 6.92483592e-06],
    [3.79485136e-01, 5.23798391e-02, 1.79097536e-02, 1.50869373e-03, 5.49193421e-06, 1.80390881e-04, -5.59831874e-06],
    [5.44284192e-01, 1.35437274e-01, 8.97205488e-04, 1.29810635e-03, 1.33243917e-06, 4.15055511e-04, 4.32625639e-07],
    [5.19558117e-01, 1.03780001e-01, 3.38571807e-03, 6.01567441e-04, -7.48199935e-07, -1.40403977e-04, -4.21020678e-07],
    [0.73996331, 0.34283568, 0.19079798, 0.06287306, 0.00633353, 0.0331251, -0.00270317],
    [4.12277593e-01, 6.14852249e-02, 4.51063234e-05, 4.14472974e-05, 1.78855138e-09, 7.62670685e-06, -1.12774941e-10],
    [4.57194069e-01, 7.96740412e-02, 5.59728600e-03, 9.01552985e-04, -3.01233001e-07, -3.84320096e-05, -2.00270802e-06],
    [2.06150459e-01, 1.47415876e-02, 4.69286118e-09, 3.50242886e-11, -8.60566804e-21, -4.19067688e-12, -1.12946039e-20]]


class plamka:
    def __init__(self, image):
        self.image = np.array(image)
        self.gray_image = color.rgb2gray(image)
        self.altered_image = color.rgb2gray(image)

    def process(self):
        '''bardzo brzydka funkcja z dużą ilością komentarzy, bo eksperymentuję, ciągle je dodaję lub usuwam'''
        seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
        if seed > 70:
            print("size too big, returning! {}".format(seed))
            return
        self.mark_colors()  # puts white wherever it thinks there is a color
        self.open(1)  # normal opening on those colors to blurr out any letters etc.
        self.find_contours()  # find all contours
        self.contour = self.get_biggest_contour()  # if there was a colorfull background
        # (see img 9 with this line commented),
        # you choose the biggest white contour
        self.altered_image = morphology.convex_hull_object(self.altered_image, 8)
        self.image, self.altered_image = self.trim_to_mask(source=self.image, mask=self.convex(self.altered_image))
        # self.show_histogram(image=self.image, label2="histogram1", bins = 256)
        self.altered_image = self.erase_colors(0.0)
        # self.show_histogram(image=self.altered_image, label2="histogram1", bins=256)
        # self.show('erase_colors')
        ''' go threw all some of the possibilities to find all numbers! (10,40) (5,90) etc'''
        p5, p95 = np.percentile(self.altered_image, (10, 40))
        self.altered_image = exposure.rescale_intensity(self.altered_image, in_range=(p5, p95))
        # self.show_histogram(image=self.altered_image, label2="rescale intensity", bins=256)
        # self.show('rescale intensity')
        '''OTSU ITADAKIMASU!'''
        threshold = filters.threshold_otsu(self.altered_image)
        self.altered_image = self.altered_image > threshold
        # self.find_contours()
        # self.show_histogram(image=self.altered_image, label2="rescale intensity", bins=256)
        self.altered_image = morphology.opening(self.altered_image, morphology.disk(2))
        # Gaussian
        self.altered_image = filters.gaussian(self.altered_image, sigma=np.sqrt(seed) / 6)
        ## label everything and check for numbers
        self.read_numbers(source=self.altered_image)

        pass

    def read_numbers(self, **kwargs):
        options = {
            'source': self.altered_image,
        }
        options.update(kwargs)

        image = options['source']

        self.label_img = measure.label(image, neighbors=4)
        self.regions = measure.regionprops(self.label_img)

        big_area = self.regions[0].convex_area
        for region in self.regions:
            if big_area < region.convex_area:
                big_area = region.convex_area

        for region in self.regions:
            if (region.convex_area == big_area):
                print(region.bbox)
                print(region.moments_hu)
                # print(region.moments_normalized)
                print(".")

    def compare_hu(self, moment, **kwargs):
        options = {
            'example_list': MOMENTS_HU,
            'distance': True
        }
        options.update(kwargs)

        examples = options['example_list']
        distance_opt = options['distance_opt']
        # result = [ max[:,i] for i in range(len(examples[0])) ] # LIST COMPREHENSION YAY!
        result = ([np.sqrt(sum(moment - max[i, :])) for i in range(len(examples))])
        for i, value in enumerate(result):
            print(i, value)

    def trim_to_mask(self, **kwargs):
        options = {
            'source': self.image,
            'mask': self.altered_image
        }
        options.update(kwargs)

        image = options['source']
        altered_image = options['mask']

        region = self.choose_mask(altered_image)
        print(region.bbox)
        altered_image = region.filled_image
        image = self.trim(image, region.bbox)
        return image, altered_image

    def choose_mask(self, image=None):
        if image is None:  # default argument
            image = self.altered_image

        self.label_img = measure.label(image, neighbors=8)
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

    # should wrap in decorator!
    def open(self, alpha=2.5):
        try:
            # self.altered_image = np.array(self.image[:,:,1])
            seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
            print(seed)
            assert seed < 50  # na na na lag proof!

            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / alpha))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / alpha))
            # self.altered_image = filters.gaussian(self.altered_image,sigma = seed)
        except AssertionError as e:
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / (alpha ** 2)))
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
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def mark_colors(self, threshold=0.25):
        for i, row in enumerate(self.image):
            for j, rgb in enumerate(row):
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                grey = (int(r) + int(g) + int(b)) / 3
                distance = abs(r - grey) + abs(g - grey) + abs(b - grey)
                if distance / 255 > threshold:  # image is 0..255 although altered is 0..1
                    self.altered_image[i][j] = 1
                else:
                    self.altered_image[i][j] = 0

    def find_contours(self):
        threshold = 0.7  # powinno zależeć od wartości kolorów w obrazku, a nie być ustalane na sztywno
        connected = 'low'
        self.contours = measure.find_contours(self.altered_image, level=threshold, fully_connected=connected)

    def convex(self, image=None):
        if image is None:
            return morphology.convex_hull_image(image)
        else:
            return morphology.convex_hull_image(self.altered_image)

    def get_biggest_contour(self):
        max_area = 0
        result = None
        for contour in self.contours:
            area = (max(contour[:, 0]) - min(contour[:, 0])) * \
                   (max(contour[:, 1]) - min(contour[:, 1]))
            if max_area < area:
                max_area = area
                result = contour
        assert not (result is None)

        return result

    def image_X_mask(self, **kwargs):
        options = {
            'source': self.image,
            'mask': self.altered_image,
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
                    image[i][j] = [pixel[0] * altered_image[i][j], \
                                   pixel[1] * altered_image[i][j], \
                                   pixel[2] * altered_image[i][j]]
        return image

    '''TODO v'''

    def binary_v(self, **kwargs):
        options = {
            'image': self.image,
            'threshold': 0.05,
        }
        options.update(kwargs)

        image = options['image']
        threshold = options['threshold']

    def show_histogram(self, **kwargs):
        options = {
            'image': self.altered_image,
            'bins': 256,
            'label1': "image",
            'label2': "histogram"

        }
        options.update(kwargs)

        image = options['image']
        bins = options['bins']
        label1 = options["label1"]
        label2 = options["label2"]

        histogram = exposure.histogram(image, bins)

        self.fig, self.plots = plt.subplots(1, 2)
        self.plots[0].imshow(image, cmap='gray')
        self.plots[1].plot(histogram[1], histogram[0], linewidth=2, zorder=1)
        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[0].set_title(label1, fontsize=10)
        self.plots[1].set_title(label2, fontsize=10)
        plt.show(block=True)

    def erase_colors(self, threshold=0.4, **kwargs):
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
                grey = (int(r) + int(g) + int(b)) / 3
                distance = abs(r - grey) + abs(g - grey) + abs(b - grey)
                if distance / 255 >= threshold:  # image is 0..255 although altered is 0..1
                    grey = max(rgb)
                    img[i][j] = grey / 255
        return img

    def erase_colors_hsv(self, **kwargs):
        options = {
            'image': self.image,
            'threshold': 0.4
        }
        options.update(kwargs)

        image = color.rgb2hsv(options['image'])
        threshold = options['threshold']

        img = np.ones((len(image), len(image[0])))
        for i, row in enumerate(image):
            for j, pixel in enumerate(row):
                h, s, v = pixel
                # if v >= threshold:  # image is 0..255 although altered is 0..1
                img[i][j] = v * (1 - s)
        return img

    def show(self, text='altered'):
        self.fig, self.plots = plt.subplots(1, 2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        if SHOW_CONTOURS:
            for contour in self.contours:
                self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
                # self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
                #  self.plots[1].plot(self.contour[:, 1], self.contour[:, 0], linewidth=2, zorder=1)

        ####
        self.label_img = measure.label(self.altered_image, neighbors=4)
        self.regions = measure.regionprops(self.label_img)

        for region in self.regions:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            self.plots[1].add_patch(rect)
        ####


        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal", fontsize=10)
        self.plots[1].set_title(text, fontsize=10)

        plt.show(block=True)

    def save(self, **kwargs):  # or self.__class__.__name__
        options = {
            'filename': "figure_" + type(self).__name__
        }
        options.update(kwargs)
        filename = options['filename']
        # io.imsave(filename,self.altered_image) # coś nie działa

        self.fig, self.plots = plt.subplots(1, 2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal", fontsize=10)
        self.plots[1].set_title("altered", fontsize=10)
        ''' upewnić się, że ten folder istnieje!!!!'''
        filename = "finals/" + filename
        self.fig.savefig(filename)
        print(filename + " has been saved")


def get_image(path, asgrey=True, _flatten=False):
    print("loading image " + path)
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


if __name__ == "__main__":
    images = "pictures_small/plamka"
    images = "example_numbers/"
    '''
    for i in range(3,4):
        #i=random.randint(1,14) # choose one of 14 images randomly
        #i = 2 #to chooose specyfic image
        image = get_image(images+str(i)+".jpg", False)
    #image = get_image("pictures_small/img (7).jpg", False)
        Plamka = plamka(image)
        Plamka.process()
        #Plamka.save(filename="try01/test01_"+str(i)+".jpg")
        Plamka.show('final')'''
    for i in range(0, 11):
        # i=random.randint(1,14) # choose one of 14 images randomly
        # i = 2 #to chooose specyfic image
        image = get_image(images + str(i) + "_3.jpg", False)
        # image = get_image("pictures_small/img (7).jpg", False)
        Plamka = plamka(image)
        # Plamka.process()
        Plamka.read_numbers()
        # Plamka.save(filename="try01/test01_"+str(i)+".jpg")
        Plamka.show('final')
