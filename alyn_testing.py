from alyn.deskew import Deskew
d = Deskew(
    input_file='image3.jpg',
    display_image='preview the image on screen',
    output_file='image3_transformed.jpg',
    r_angle=0.0)
d.run()