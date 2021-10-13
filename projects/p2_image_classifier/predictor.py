
import numpy as np
from PIL import Image
from process_image import process_image

def predict(image_path, model, top_k):
    #prepare the image data
    im = Image.open(image_path)
    image = np.asarray(im) #convert to numpy array
    processed_image = process_image(image) #call image preprocessing function
    processed_image = np.expand_dims(processed_image, axis = 0) # add extra dimension to meet expected input shape

    #run prediction model
    predicted = model.predict(processed_image)

    # get the top_k probability
    proba = - np.partition(-predicted[0], top_k)[:top_k]
    classes = np.argpartition(-predicted[0], top_k)[:top_k]
    return proba, list(classes)
