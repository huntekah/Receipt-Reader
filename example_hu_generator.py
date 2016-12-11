from skimage import io, measure
import os.path

DIRECTORY = "example_numbers/multiple_examples/"
names = ["0","1","2","3","4","5","6","7","8","9","comma","stain"]


def get_image(path, asgrey=True, _flatten=False):
    return io.imread(path, as_grey=asgrey, flatten=_flatten)

def get_hu(image):
    m = measure.moments(image)
    mu = measure.moments_central(image, m[0, 1] / m[0, 0] , m[1, 0] / m[0, 0])
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)
    return hu

def show(hu):
    print("[", end=" ")
    print(hu, end = "")
    print(" ]", end = ",\n")

def trim(image):
    threshold = 0.4
    min_row = None
    min_col = None
    max_row = None
    max_col = None
    for i, row in enumerate(image):
        if min_row is None:
            if min(row) < threshold:
                min_row = i

        if max_row is None:
            if min(row) < threshold:
                max_row = i
        elif max_row < i and min(row) < threshold:
            max_row = i

        for j, pixel in enumerate(row):

            if min_col is None:
                if pixel < threshold:
                    min_col = j
            elif pixel < threshold and j < min_col:
                min_col = j

            if max_col is None:
                if pixel < threshold:
                    max_col = j
            elif max_col < j and pixel < threshold:
                max_col = j

    assert not (min_row is None)
    assert not (max_row is None)
    assert not (min_col is None)
    assert not (max_col is None)

    return image[min_row : (max_row+1), min_col:(max_col+1)]



if __name__ == "__main__":

    print("MOMENTS_HU = [\n", end=" ")

    for name in names:
        i = 0
        fname = str(DIRECTORY + name + "_" + str(i) + ".jpg")
        while os.path.exists(fname):
            #print(fname)
            i = i + 1
            image = get_image(fname, _flatten=True)
            image = trim(image)
            image = 1-image
            hu = get_hu(image)
            string = ', '.join([str(element) for element in hu])
            string+=(', "'+name+'"')
            show(string)
            fname = str(DIRECTORY + name + "_" + str(i) + ".jpg")

    print(" ]", end = "\n")
