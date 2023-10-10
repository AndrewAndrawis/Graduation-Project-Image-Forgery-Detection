from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn import svm
import pickle
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from skimage.util import view_as_windows
from scipy.fftpack import fft, dct
import tensorflow
from tensorflow import keras
from keras import applications
from efficientnet.tfkeras import EfficientNetB4
from keras import layers, Model
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall, FalseNegatives, FalsePositives, TrueNegatives, \
    TruePositives, AUC
from keras.applications.imagenet_utils import decode_predictions

def get_patches(image_mat):
    """
    Extract patches rom an image
    :param image_mat: The image as a matrix
    :param stride: The stride of the patch extraction process
    :returns: The patches
    """
    stride=8 #stride is same as window's breadth so that it gives non-overlapping blocks.
    window_shape = (8, 8)
    image_mat=np.array(image_mat)
    
    windows = view_as_windows(image_mat, window_shape, step=stride)
#     print('windows shape:',windows.shape)

    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
#             print("window shape: ",windows[m][n].shape)
            patches += [windows[m][n]]
    return patches

def std_and_ones(type_of_sub_image_blocks):
    ac_dct_stack=[]
    number_of_ones=[]

    for block in type_of_sub_image_blocks:
        dct_block = dct(block, type=2, n=None, axis=-1, norm=None, overwrite_x=False)
        dct_block_row = dct_block.flatten() # 2d dct array to 1d row array.
        ac_dct = dct_block_row[1:] # only AC component, removing the first DC comp.
        ac_dct_stack.append(ac_dct)

    ac_dct_stack=np.asarray(ac_dct_stack) #1536X63
    ac_dct_stack=ac_dct_stack.T # 63X1536

#     print("AC stacked shape: ", ac_dct_stack.shape)

    ac_dct_std = np.std(ac_dct_stack, axis=1) # row wise standard-deviation.

    for i in range(ac_dct_stack.shape[0]):
        count_one=0
        for j in range(ac_dct_stack.shape[1]):
            if(ac_dct_stack[i][j]>0):   # row wise counting number of ones.
                count_one+=1
        number_of_ones.append(count_one)

    number_of_ones=np.asarray(number_of_ones)
    
    return(ac_dct_std, number_of_ones)

def feature_sub_image(sub_image):
    sub_image_blocks = get_patches(sub_image) #Gives the 8x8 patches/blocks of sub_image.

    sub_image_cropped = sub_image[4:,4:] #removing 4 rows and 4 cols.
    sub_image_cropped_blocks = get_patches(sub_image_cropped)

    STD_full_image, ONE_full_image = std_and_ones(sub_image_blocks)
    STD_cropped_image, ONE_cropped_image = std_and_ones(sub_image_cropped_blocks)

    #             print("STD_full image shape: ",STD_full_image.shape)
    #             print("one_full image shape: ",ONE_full_image.shape)
    #             print("STD_crop image shape: ",STD_cropped_image.shape)
    #             print("One_crop image shape: ",ONE_cropped_image.shape)
    
    #63x4 stacked F-sub-image
    F_sub_image=np.column_stack((STD_full_image, ONE_full_image, STD_cropped_image, ONE_cropped_image))
    
    F_sub_image_flat=F_sub_image.T.flatten() #column wise flattening, 63*4=252 features
    return(F_sub_image_flat)

def feature_extraction(path_to_folder, class_label):
    data_list=[]
    count = 0
    for file_name in os.listdir(path_to_folder):
        if (count != 0 and count % 300 == 0):
            print("No. of images done: ", count)
        path_to_img = os.path.join(path_to_folder,file_name)
        img = cv2.imread(path_to_img)
        
        if np.shape(img) == ():
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) #changing to YCrCb color space.
        img_y = img[:,:,0] # the Y channel only.
        img_cr = img[:,:,1] # the Cr channel only.
        img_cb = img[:,:,2] # the Cb channel only.
        
        Fy = feature_sub_image(img_y)
        Fcr = feature_sub_image(img_cr)
        Fcb = feature_sub_image(img_cb)
#         print("fy shape: ",Fy.shape)
#         print("fcr shape: ",Fcr.shape)
#         print("fcb shape: ",Fcb.shape)
        
        final_feature = np.concatenate((Fy, Fcb, Fcr), axis=None) #63*4*3=756 flattened features.
#         print("final feature shape: ",final_feature.shape)
        
        final_feature=list(final_feature)
        final_feature.insert(0,file_name)
        final_feature.insert(1,class_label)
        data_list.append(final_feature)
        
        
    return(data_list)

#newima = self.feature_extraction(fname,0)
#ndf = pd.DataFrame(newima)
#output_name = 'test_features'
#ndf.to_csv(output_name, index=False)
#ndf.drop(columns = ndf.columns[0], axis = 1, inplace= True)
#ndf.drop(columns = ndf.columns[1], axis = 1, inplace= True)
 #with open('static\shilpla_model1.pkl', 'rb') as f:
            #trained_model = pickle.load(f) 
#result = trained_model.predict(ndf)




app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mlupload', methods = ['GET', 'POST'])
def ml_upload():
   if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if filename == "":
            return render_template('index.html', ml_prediction = 'Please upload an image.')

        file.save(os.path.join(app.config['UPLOAD'], filename))
        newima = feature_extraction(app.config['UPLOAD'],0)
        ndf = pd.DataFrame(newima)
        output_name = 'test_features'
        ndf.to_csv(output_name, index=False)
        ndf.drop(columns = ndf.columns[0], axis = 1, inplace= True)
        ndf.drop(columns = ndf.columns[1], axis = 1, inplace= True)
        with open('C:\Users\andre\Desktop\Grad 1\my_trained_model8.pkl', 'rb') as f:
            trained_model = pickle.load(f) 
        result = trained_model.predict(ndf)
        os.remove(os.path.join(app.config['UPLOAD'], filename))
        if 1 in result:
            print(result)
            return render_template('index.html', ml_prediction = 'Image is forged.')
        else:
            return render_template('index.html', ml_prediction = 'Image is authentic.')

@app.route('/r5upload', methods = ['GET', 'POST'])
def r5_upload():
   if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if filename == "":
            return render_template('index.html', dl_prediction = 'Please upload an image.')
        file.save(os.path.join(app.config['UPLOAD'], filename))
        
        img = cv2.imread(os.path.join(app.config['UPLOAD'], filename))
        imgresize = cv2.resize(img,(224,224))
        #cv2.imwrite(os.path.join(app.config['UPLOAD'],'imgres.jpg'), imgres)
        imgres = imgresize.astype(np.float32) / 255.0
        imgpred = np.expand_dims(imgres, axis=0)
        
        #imgs = imgres.reshape((1,) + imgres.shape)
        #print(imgres.shape)
        spmodel = tensorflow.keras.models.load_model('"C:\Users\andre\Desktop\conda\grad\sp\Final_model_Resnet50 again.h5"')

        
        spmodel.compile(Adam(learning_rate=0.001), loss=BinaryCrossentropy(),
                 metrics=[AUC(), BinaryAccuracy(), Precision(), Recall(), FalseNegatives(),
                          FalsePositives(), TrueNegatives(), TruePositives() ])
        spresult = spmodel.predict(imgpred)[0]


        print("Done")
        print(spresult)
 
        if spresult > 0.5:

            return render_template('index.html', r5_prediction = 'Image is forged(Spliced).' )
        else:
            os.remove(os.path.join(app.config['UPLOAD'], filename))
            return render_template('index.html', r5_prediction = 'Image is authentic.')
        
@app.route('/v3upload', methods = ['GET', 'POST'])
def v3_upload():
   if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if filename == "":
            return render_template('index.html', dl_prediction = 'Please upload an image.')
        file.save(os.path.join(app.config['UPLOAD'], filename))
        
        img = cv2.imread(os.path.join(app.config['UPLOAD'], filename))

        cmmodel = tensorflow.keras.models.load_model('static/inception V3 copy move_4.h5')

        

        imgcm = cv2.resize(img,(299,299))
        imgcmpred = np.expand_dims(imgcm, axis=0)
        cmresult = cmmodel.predict(imgcmpred)

        print("Done")

        print(cmresult.argmax(axis=1))
        print(cmresult[0][0])
        print(cmresult[0])
        print(cmresult)
        if cmresult > 0.5:
            return render_template('index.html', v3_prediction = 'Image is forged(Copy-move).' )
        else:
            os.remove(os.path.join(app.config['UPLOAD'], filename))
            return render_template('index.html', v3_prediction = 'Image is authentic.')


@app.route('/aiupload', methods = ['GET', 'POST'])
def ai_upload():
   if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        if filename == "":
            return render_template('index.html', ai_prediction = 'Please upload an image.')
        file.save(os.path.join(app.config['UPLOAD'], filename))
        
        img = cv2.imread(os.path.join(app.config['UPLOAD'], filename))
        img = cv2.resize(img,(256,256))
        test_image_arr = tensorflow.keras.preprocessing.image.img_to_array(img)
        test_image_arr = np.expand_dims(img, axis=0)
        test_image_arr = test_image_arr/255.

        aimodel = tensorflow.keras.models.load_model('C:\Users\andre\Desktop\conda\grad\ai gen\ai_gen_model.h5')
        airesult = aimodel.predict(test_image_arr)
        

     

        print("Done")
        print(airesult)
        print(airesult[0][0])
     
        
        if airesult[0][0] > 0.5:

            return render_template('index.html', ai_prediction = 'Face is AI generated.' )
        else:
            os.remove(os.path.join(app.config['UPLOAD'], filename))
            return render_template('index.html', ai_prediction = 'Face is real.')
        
        



if __name__ == "__main__":
    app.run(debug=True)


