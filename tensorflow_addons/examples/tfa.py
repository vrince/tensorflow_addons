
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa

#from tensorflow_addons.image import threshold

threshold_images = tfa.image.image_threshold(
    [[[1.2, 2.5], [0.2, 4.9]]], [1, 2, 3])
    
print(threshold_images)
