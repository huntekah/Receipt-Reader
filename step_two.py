import paragon_proscessing
from skimage import data, io, filters, feature, morphology, measure, exposure, color, util
# from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from CART import cart, MOMENTS_HU

SHOW_CONTOURS = False
CART_DEPTH = 9
FIND_CONTORUS_THRESHOLD = 0.7
MARK_COLORS_THRESHOLD = 0.25
ALPHA = 2.5
PERCENTILE_0 = 0.1
PERCENTILE_1 = 0.4
ERASE_COLORS_THRESHOLD = 0.4

class plamka:
    def __init__(self, image):
        self.image = np.array(image)
        self.gray_image = color.rgb2gray(image)
        self.altered_image = color.rgb2gray(image)

    def process(self):
        '''Main function that lead's threw the image altering process'''
        seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
        if seed > 70:
            print("size too big, returning! {}".format(seed))
            return
        self.mark_colors()  # puts white wherever it thinks there is a color
        self.open(1)  # normal opening on those colors to blurr out any letters etc.
        self.find_contours()  # find all contours
        self.contour = self.get_biggest_contour()  # if there was a colorfull background
        # you choose the biggest white contour
        self.altered_image = morphology.convex_hull_object(self.altered_image, 8)
        self.image, self.altered_image = self.trim_to_mask(source=self.image, mask=self.convex(self.altered_image))
        self.altered_image = self.erase_colors(0.0)
        ''' go threw some of the possibilities to find all numbers! (10,40) (5,90) etc'''
        p5, p95 = np.percentile(self.altered_image, (PERCENTILE_0, PERCENTILE_1))
        self.altered_image = exposure.rescale_intensity(self.altered_image, in_range=(p5, p95))
        '''OTSU ITADAKIMASU!'''
        threshold = filters.threshold_otsu(self.altered_image)
        self.altered_image = self.altered_image > threshold
        self.altered_image = morphology.opening(self.altered_image, morphology.disk(2))
        ## label everything and check for numbers
        self.read_numbers(source=self.altered_image)

        pass

    def read_numbers(self, **kwargs):
        '''Uses CART from sklearn to effectively read numbers. Omits stains and commas.'''
        options = {
            'source': self.altered_image,
            'method': 'CART'
        }
        options.update(kwargs)

        image = options['source']
        image = 1 - image
        method = options['method']

        ######################## CART ########################
        if method.upper() == 'CART':
            self.label_img = measure.label(image, neighbors=4, background=0, connectivity=1)
            self.regions = measure.regionprops(self.label_img)

            self.final_data = []

            Cart = cart(CART_DEPTH)

            for region in self.regions:
                #print(region.bbox)
                #print(region.moments_hu)
                size = (region.bbox[1] - region.bbox[0]) * (region.bbox[3] - region.bbox[2])
                average_color = self.average_color(self.trim(self.image, bbox=region.bbox))  # !on color image
                number = Cart.predict( (region.moments_hu).reshape(1, -1) )
                proba = Cart.predict_proba( (region.moments_hu).reshape(1, -1) )
                if number != 'stain' and number != 'comma':
                    self.final_data.append((number, region.bbox, size, region.centroid, average_color, proba))

            self.show1(image=image, text="All bbox'es", bbox_iterable=[region[1] for region in self.final_data], color = 'blue')
            for number in self.final_data:
                print("This may be: ", number[0][0], "\nwith color: ", number[4])
                #_color = rgb2hex(number[4])
                _color = 'red'
                self.show1(image=image, text=str(number[0]), bbox=number[1], color = _color )

    def num_of_matches(self, **kwargs):
        options = {
            'list': self.final_data,
            'size': 0,
            'tolreance': 0.3,
            'bbox': None,  # no default value. coudl be ommited, but want to ephasize that there's such an option
        }
        options.update(kwargs)

        list = options['list']
        size = options['size']
        tolerance = options['tolerance']
        bbox = options['bbox']

        min_size = size * (1 - tolerance)
        max_size = size * (1 + tolerance)
        count = 0

        for element in list:
            if min_size <= element[2] <= max_size:
                count = count + 1

        return count

    '''Remake it into K-NN'''
    def compare_hu(self, moment, **kwargs):
        '''Depricated version of KNN. DO NOT USE!'''
        options = {
            'example_list': MOMENTS_HU,
            'verbose': False
        }
        options.update(kwargs)

        examples = options['example_list']
        verbose = options['verbose']
        distance = []
        for i in range(len(examples)):
            tmp = []
            for j in range(len(examples[i])):
                tmp.append(moment[j] - examples[i][j])
            distance.append(tmp)
        result = np.sqrt([sum(distance[i]) ** 2 for i in range(len(distance))])
        normalize(result)

        # now find percentage assigned to the list! :)
        result = 1 - result
        sum_r = sum(result)
        result = [100 * (i / sum_r) for i in result]

        if verbose:
            print("dystans")
            print(distance)
            for i, value in enumerate(result):
                print(i, "\t\t", value, "%")

        return result

    def average_color(self, image):
        '''returns average color in given image'''
        r, g, b = 0, 0, 0
        count = len(image)*len(image[0])
        for row in image:
            for pixel in row:
                if len(pixel) == 3:
                    r = r + pixel[0]
                    g = g + pixel[1]
                    b = b + pixel[2]
                elif len(pixel) == 1:
                    r = r + pixel[0]
                    g = g + pixel[0]
                    b = b + pixel[0]
                else:
                    raise Exception("Wrong Input provided")

        return r / count, g / count, b / count

    def trim_to_mask(self, **kwargs):
        '''Returns image that has been trimmed to the given mask.
        It chooses biggest region (see choose_mask) and trims
        image to the bbox of that bigest region'''
        options = {
            'source': self.image,
            'mask': self.altered_image,
            'verbose' : False
        }
        options.update(kwargs)

        image = options['source']
        altered_image = options['mask']
        verbose = options['verbose']

        region = self.choose_mask(altered_image)
        if verbose:
            print(region.bbox)
        altered_image = region.filled_image
        image = self.trim(image, region.bbox)
        return image, altered_image

    def choose_mask(self, image=None):
        '''Choses region with biggest area as the best fit to be
         the area of interest in the receipt'''
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
        ''' prints values x,y alongside of the given contour'''
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

    #FUTURE: should wrap in decorator!
    def open(self, alpha=ALPHA):
        '''performs normal opening operation'''
        try:
            seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
            print(seed)
            assert seed < 50  # na na na lag proof!

            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / alpha))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / alpha))
        except AssertionError as e:
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def close(self, alpha=ALPHA):
        '''performs normal closing operation'''
        try:
            seed = (min(len(self.altered_image), len(self.altered_image[0])) ** 2) / 10000
            print(seed)
            assert seed < 50  # to prevent lags

            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / alpha))
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / alpha))
        except AssertionError as e:
            self.altered_image = morphology.opening(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            self.altered_image = morphology.closing(self.altered_image, morphology.disk(seed / (alpha ** 2)))
            print("seed size too big: {}! \nresize image please!".format(seed))
            print(e)

    def mark_colors(self, threshold=MARK_COLORS_THRESHOLD):
        '''marks color that fit the given threshold which is distance from average grey (0..1) = (almost_grey..super_colorfull'''
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

    def find_contours(self, threshold = FIND_CONTORUS_THRESHOLD):
        '''finds contours on the image'''
        connected = 'low'
        self.contours = measure.find_contours(self.altered_image, level=threshold, fully_connected=connected)

    def convex(self, image=None):
        '''returns convex hull version of the image'''
        if not(image is None):
            return morphology.convex_hull_image(image)
        else:
            return morphology.convex_hull_image(self.altered_image)

    def get_biggest_contour(self):
        '''returns biggest contour'''
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
        '''[binary] multiplicatin = (image) x (mask)'''
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


    def show_histogram(self, **kwargs):
        '''shows histogram of given image'''
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

    def erase_colors(self, threshold=ERASE_COLORS_THRESHOLD, **kwargs):
        '''try's its best to erase all colors'''
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
        '''does color deletion using info from hsv scale'''
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
                img[i][j] = v * (1 - s)
        return img

    def show1(self,image=None, text='altered', bbox=[0, 0, 0, 0], color = 'red', bbox_iterable = None, oryginal_image = None):
        '''Used to show the bbox around certain stain/number (good for showing results)'''
        if image is None:
            image = self.altered_image
        if oryginal_image is None :
            oryginal_image = self.image
        self.fig, self.plots = plt.subplots(1, 2)
        self.plots[0].imshow(oryginal_image)
        self.plots[1].imshow(image, cmap='gray')

        if SHOW_CONTOURS:
            for contour in self.contours:
                self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)

        ####
        if not(bbox_iterable is None):
            for bbox in bbox_iterable:
                minr, minc, maxr, maxc = bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor=color, linewidth=2)
                self.plots[1].add_patch(rect)
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=color, linewidth=2)
        self.plots[1].add_patch(rect)
        ####


        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                  labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal", fontsize=10)
        self.plots[1].set_title(text, fontsize=10)

        plt.show(block=True)

    def show(self, text='altered'):
        '''basic show function'''
        self.fig, self.plots = plt.subplots(1, 2)
        self.plots[0].imshow(self.image)
        self.plots[1].imshow(self.altered_image, cmap='gray')

        if SHOW_CONTOURS:
            for contour in self.contours:
                self.plots[0].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
                # self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)
                #  self.plots[1].plot(self.contour[:, 1], self.contour[:, 0], linewidth=2, zorder=1)

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
    '''used to load an image'''
    print("loading image " + path)
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


def normalize(_list):
    '''used for list normalization'''
    r_list = _list[:]
    for i, element in enumerate(_list):
        if isinstance(element, list):
            r_list[i] = normalize(element)
        else:
            r_list[i] = (element - min(_list)) / (max(_list) - min(_list))
    _list = r_list
    return _list

def rgb2hex(rgb):
    '''rgb to hsv conversion'''
    r, g, b = rgb
    if isinstance(r, float):
        r = int(r * 255)
    if isinstance(r, float):
        g = int(g * 255)
    if isinstance(r, float):
        b = int(b * 255)
    r = int(max(0, min(r, 255)))
    g = int(max(0, min(g, 255)))
    b = int(max(0, min(b, 255)))
    "#{0:02x}{1:02x}{2:02x}".format(255 - r, 255 - g, 255 - b)

if __name__ == "__main__":
    images = "pictures_small/plamka"

    for i in range(1, 14):
        # i=random.randint(1,14) # choose one of 14 images randomly
        image = get_image(images + str(i) + ".jpg", False)
        Plamka = plamka(image)
        Plamka.process()
        # Plamka.save(filename="try01/test01_"+str(i)+".jpg")
        Plamka.show('final')

