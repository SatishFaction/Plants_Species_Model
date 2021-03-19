from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)
import os
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image
#import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
clas=pd.read_csv(r'C:\Users\hp\Desktop\Plants model\classname.csv')
class_names=list(clas.Class_Name)

# define a Flask app

MODEL = load_model(r'C:\Users\hp\Desktop\Deploy_clone\Deployed_ML_model_for_image_classifier\Plant_model_5_Epoch_new.h5',custom_objects={'KerasLayer': hub.KerasLayer})
#graph = tf.get_default_graph()

print('Successfully loaded Plants Species model...')
print('Visit http://127.0.0.1:5000')

def model_predict(img_path):
    '''
        helper method to process an uploaded image
    '''
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image=image/255.0
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    preds = MODEL.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(preds[0], axis=-1)
    class_nam=class_names[predicted_class]
    #global graph
    #with graph.as_default():
    return class_nam

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # get the file from the HTTP-POST request
        f = request.files['file']        
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        
        # make prediction about this image's class
        preds = model_predict(file_path)
        
        #pred_class = decode_predictions(preds, top=10)
        result = preds#str(pred_class[0][0][1])
        #print('[PREDICTED CLASSES]: {}'.format(pred_class))
        #print('[RESULT]: {}'.format(result))
        print(result)
        
        return result
    
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True,use_reloader=False)
