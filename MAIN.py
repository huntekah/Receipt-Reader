from step_two import plamka, get_image
from paragon_proscessing import img_processing

if __name__ == "__main__":
    images = "pictures_small/img ("
    '''images = "example_numbers/"'''

    for i in range(1,17):

        paragon = img_processing(images+str(i)+").jpg")
        image = paragon.process()
    #image = get_image("pictures_small/img (7).jpg", False)
        Plamka = plamka(image)
        Plamka.process()
        #Plamka.save(filename="try01/test01_"+str(i)+".jpg")
        Plamka.show('final')