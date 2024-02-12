import numpy as np
import cv2
import insightface

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


print('insightsface', insightface.__version__)
print('numpy', np.__version__)

app = FaceAnalysis(name='buffalo_l')

app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download=True,
                                          download_zip=False)


def swap(image1, app, swapper, image2, output_path1, output_path2):

    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0] 

    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_ = swapper.get(img1_, face1, face2, paste_back=True)
    img2_ = swapper.get(img2_, face2, face1, paste_back=True) 

    cv2.imwrite(output_path1, img1_)
    cv2.imwrite(output_path2, img2_)
    return img1_, img2_

output_image1 = 'output_image1.png'
output_image2 = 'output_image2.png'

# Corrected order of arguments in the function call
output_image = swap('phere_bride.png', app, swapper, 'phere_groom.png', output_image1, output_image2)