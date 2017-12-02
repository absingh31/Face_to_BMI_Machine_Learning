#!/usr/bin/python

import sys
import cv2
import dlib
import numpy
import pandas as pd
import pickle
import sklearn

# INPUT_PATH = 'bmiapp-data-2017-04-25/'
INPUT_PATH = 'INPUT/'

def get_landmarks(im):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cascade_path='haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)

    rects = cascade.detectMultiScale(im, 1.3,5)

    if len(rects) > 0:
        x,y,w,h =rects[0]
        rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    else:
        return 0

def predict(image_name_list):
    collect_landmarks = []

    for image_name in image_name_list:
        img=cv2.imread(INPUT_PATH+image_name)
        landmark_list = get_landmarks(img)
        if isinstance(landmark_list,int) == False:
            collect_landmarks.append([image_name,landmark_list.tolist()])

    inmate_2d_list = []

    num_inmates = len(collect_landmarks)

    for each_inmate in range(0,num_inmates):
        inmate_1d_list = []
        inmate_id_landmark = collect_landmarks[each_inmate][0]
        inmate_1d_list.append(inmate_id_landmark)
        for each_landmark in range(0,68):
            landx = collect_landmarks[each_inmate][1][each_landmark][0]
            inmate_1d_list.append(landx)
            landy = collect_landmarks[each_inmate][1][each_landmark][1]
            inmate_1d_list.append(landy)
        inmate_2d_list.append(inmate_1d_list)


    df_col_names = ['image_name',
    'landx0',
    'landy0',
    'landx1',
    'landy1',
    'landx2',
    'landy2',
    'landx3',
    'landy3',
    'landx4',
    'landy4',
    'landx5',
    'landy5',
    'landx6',
    'landy6',
    'landx7',
    'landy7',
    'landx8',
    'landy8',
    'landx9',
    'landy9',
    'landx10',
    'landy10',
    'landx11',
    'landy11',
    'landx12',
    'landy12',
    'landx13',
    'landy13',
    'landx14',
    'landy14',
    'landx15',
    'landy15',
    'landx16',
    'landy16',
    'landx17',
    'landy17',
    'landx18',
    'landy18',
    'landx19',
    'landy19',
    'landx20',
    'landy20',
    'landx21',
    'landy21',
    'landx22',
    'landy22',
    'landx23',
    'landy23',
    'landx24',
    'landy24',
    'landx25',
    'landy25',
    'landx26',
    'landy26',
    'landx27',
    'landy27',
    'landx28',
    'landy28',
    'landx29',
    'landy29',
    'landx30',
    'landy30',
    'landx31',
    'landy31',
    'landx32',
    'landy32',
    'landx33',
    'landy33',
    'landx34',
    'landy34',
    'landx35',
    'landy35',
    'landx36',
    'landy36',
    'landx37',
    'landy37',
    'landx38',
    'landy38',
    'landx39',
    'landy39',
    'landx40',
    'landy40',
    'landx41',
    'landy41',
    'landx42',
    'landy42',
    'landx43',
    'landy43',
    'landx44',
    'landy44',
    'landx45',
    'landy45',
    'landx46',
    'landy46',
    'landx47',
    'landy47',
    'landx48',
    'landy48',
    'landx49',
    'landy49',
    'landx50',
    'landy50',
    'landx51',
    'landy51',
    'landx52',
    'landy52',
    'landx53',
    'landy53',
    'landx54',
    'landy54',
    'landx55',
    'landy55',
    'landx56',
    'landy56',
    'landx57',
    'landy57',
    'landx58',
    'landy58',
    'landx59',
    'landy59',
    'landx60',
    'landy60',
    'landx61',
    'landy61',
    'landx62',
    'landy62',
    'landx63',
    'landy63',
    'landx64',
    'landy64',
    'landx65',
    'landy65',
    'landx66',
    'landy66',
    'landx67',
    'landy67']


    df_landmark = pd.DataFrame(inmate_2d_list,columns=df_col_names)

    with open('rf_model_v3.pkl', 'rb') as f:
        rf = pickle.load(f)

    bmi_data = df_landmark.copy()

    def calc_dist(x0,y0,x1,y1):
        dist = (x0-x1)**2 + (y0-y1)**2
        dist = dist ** 0.5
        return dist

    bmi_data['face_width0'] = calc_dist(bmi_data['landx0'],bmi_data['landy0'],bmi_data['landx16'],bmi_data['landy16'])
    bmi_data['face_width1'] = calc_dist(bmi_data['landx1'],bmi_data['landy1'],bmi_data['landx15'],bmi_data['landy15'])
    bmi_data['face_width2'] = calc_dist(bmi_data['landx2'],bmi_data['landy2'],bmi_data['landx14'],bmi_data['landy14'])
    bmi_data['face_width3'] = calc_dist(bmi_data['landx3'],bmi_data['landy3'],bmi_data['landx13'],bmi_data['landy13'])
    bmi_data['face_width4'] = calc_dist(bmi_data['landx4'],bmi_data['landy4'],bmi_data['landx12'],bmi_data['landy12'])
    bmi_data['face_width5'] = calc_dist(bmi_data['landx5'],bmi_data['landy5'],bmi_data['landx11'],bmi_data['landy11'])
    bmi_data['face_width6'] = calc_dist(bmi_data['landx6'],bmi_data['landy6'],bmi_data['landx10'],bmi_data['landy10'])
    bmi_data['face_width7'] = calc_dist(bmi_data['landx7'],bmi_data['landy7'],bmi_data['landx9'],bmi_data['landy9'])

    bmi_data['face_height'] = calc_dist(bmi_data['landx27'],bmi_data['landy27'],bmi_data['landx8'],bmi_data['landy8'])

    bmi_data['eye_width1'] = calc_dist(bmi_data['landx36'],bmi_data['landy36'],bmi_data['landx45'],bmi_data['landy45'])
    bmi_data['eye_width2'] = calc_dist(bmi_data['landx39'],bmi_data['landy39'],bmi_data['landx42'],bmi_data['landy42'])


    bmi_data['face_ratio2']=bmi_data['face_width2']/bmi_data['face_width1']
    bmi_data['face_ratio3']=bmi_data['face_width3']/bmi_data['face_width1']
    bmi_data['face_ratio4']=bmi_data['face_width4']/bmi_data['face_width1']
    bmi_data['face_ratio5']=bmi_data['face_width5']/bmi_data['face_width1']
    bmi_data['face_ratio6']=bmi_data['face_width6']/bmi_data['face_width1']
    bmi_data['face_ratio7']=bmi_data['face_width7']/bmi_data['face_width1']

    bmi_data['face_ratioh1']=bmi_data['face_width1']/bmi_data['face_height']
    bmi_data['face_ratioh2']=bmi_data['face_width2']/bmi_data['face_height']
    bmi_data['face_ratioh3']=bmi_data['face_width3']/bmi_data['face_height']
    bmi_data['face_ratioh4']=bmi_data['face_width4']/bmi_data['face_height']
    bmi_data['face_ratioh5']=bmi_data['face_width5']/bmi_data['face_height']
    bmi_data['face_ratioh6']=bmi_data['face_width6']/bmi_data['face_height']
    bmi_data['face_ratioh7']=bmi_data['face_width7']/bmi_data['face_height']

    bmi_data['eye_ratio1']=(bmi_data['eye_width1']-bmi_data['eye_width2'])/bmi_data['face_width0']
    bmi_data['eye_ratio2']=(bmi_data['eye_width1']-bmi_data['eye_width2'])/bmi_data['face_height']


    feature_list = [
    'face_ratio2',
    'face_ratio3',
    'face_ratio4',
    'face_ratio5',
    'face_ratio6',
    'face_ratio7',
    'face_ratioh1',
    'face_ratioh2',
    'face_ratioh3',
    'face_ratioh4',
    'face_ratioh5',
    'face_ratioh6',
    'face_ratioh7',
    'eye_ratio1',
    'eye_ratio2']                

    results = rf.predict_proba(bmi_data[feature_list])
    bmi_data['score_bmi30'] = results[:,1]

    output = 'img_BMI_prediction.csv'
    bmi_data[['image_name','score_bmi30']].to_csv(output)

    print("Prediction Output File: {}".format(output))

#--------------------------------------------------------------------------

def main():
    image_args = sys.argv[1:]
    print(image_args)
    predict(image_name_list=image_args)

if __name__ == "__main__":
    main()
