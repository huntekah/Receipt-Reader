#paragon..
from skimage import data, io, filters, feature, morphology, measure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np

class img_processing:
    def __init__(self, image):
        self.image = get_image(image,False)
        self.gray_image = get_image(image,True,True)
        self.altered_image = np.array(self.gray_image)

    def process(self):
        self.open()
        self.find_contours() # najpierw znajdź całą kartkę paragonu
        #potem konkretne plamki (albo od razu plamik)
        # przejdź po znalezionych plamkach (konturach) i znajdź tą, która może mieć ceny na sobie.
        #   np. (zobaczyć która ma dużo podobnych kolorów + jest otoczona przez biel albo jakkolwiek inaczej.)
        #   lub wziąć tą najbardziej na prawo
        # spośród pixeli wewnątrz konturu(plamki) stworzyć nowy obrazek // kontur to tylko ramka
        # wysłać ten obrazek do jakiejś funkcji foo, którą ja będę pisał :)
        # foo powinna zwracać true jeśli znalazła ceny w plamce, false jeśli nie

    def open(self):
        self.altered_image = morphology.opening(self.altered_image, morphology.disk(len(self.altered_image) / 100))
        self.altered_image = morphology.dilation(self.altered_image, morphology.disk(len(self.altered_image)/ 35))
        self.altered_image = morphology.closing(self.altered_image, morphology.disk(len(self.altered_image) / 40))
        self.altered_image = morphology.opening(self.altered_image, morphology.disk(len(self.altered_image) / 40))
        self.altered_image = morphology.erosion(self.altered_image, morphology.disk(len(self.altered_image) / 35))

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
            self.plots[1].plot(contour[:, 1], contour[:, 0], linewidth=2, zorder=1)

        self.plots[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                            labelleft='off', labelbottom='off')
        self.plots[0].set_title("oryginal",fontsize=10)
        self.plots[1].set_title("altered",fontsize=10)


        plt.show()

def hsv2rgb(h, s, v):
        h = h % 360.0  # color scale is a circle!
        s = max(min(s, 1.0), 0.0)  # limit s to range 0..1
        v = max(min(v, 1.0), 0.0)  # limit s to range 0..1
        h = h / 60
        section = h // 1
        fraction = h - section
        Mid_odd = v * (1 - s * fraction)
        Mid_even = v * (1 - s + s * fraction)
        m = v - v * s
        # v - main color
        # m - amount to match value
        # Mid_odd
        # Mid_even - its like Mid_odd, but going the other direction (used in every second section)
        if section == 0:
            return v, Mid_even, m
        elif section == 1:
            return Mid_odd, v, m
        elif section == 2:
            return m, v, Mid_even
        elif section == 3:
            return m, Mid_odd, v
        elif section == 4:
            return Mid_even, m, v
        elif section == 5:
            return v, m, Mid_odd
        return (-1, -1, -1)  # if something goes wrong, let it return those results

def rgb2hsv(r,g,b):
    #idk maby later :D
    pass


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)


if __name__ == "__main__":
    #img_processing.foo(a=7,b=3)
    #paragon = img_processing("test.jpg")
    paragon = img_processing("pictures_small/img (1).jpg")
    paragon.process()
    paragon.show()
