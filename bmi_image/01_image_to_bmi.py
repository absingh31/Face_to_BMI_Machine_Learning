import cv2
import dlib
import numpy
import pandas as pd
import pickle


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    if len(rects) > 0:
        x,y,w,h =rects[0]
        rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    else:
        return 0



collect_landmarks = []

image_name_list = [
'7CF1A043-8C3B-4616-897E-FAEF5468076D.jpeg',
'A0515CF3-F573-423E-A7FC-4336A888E47E.jpeg',
'EF8E2B2D-F3D6-40EB-8C4D-3CB5D347554B.jpeg',
'50D57451-6184-4E75-B36B-C459CAF603FC.jpeg',
'027B693E-AFDA-470D-9E34-6A888754F078.jpeg',
'A98BF5E7-E56C-41A8-9D95-25A82A144EC2.jpeg',
'04921E48-9862-4C75-B06B-AA99D8A3ED2E.jpeg',
'0FEDA0A9-7787-4033-9406-2435D3B4C874.jpeg',
'6764E148-35E6-4509-8D2F-7F4F365ADB5E.jpeg',
'49D3351C-5079-42BD-B2E8-83376743AD50.jpeg',
'AD610A9E-639D-459B-8512-1B977A5604B9.jpeg',
'51D2A651-7C69-4D68-B214-A41BABD18134.jpeg',
'8F07F1C9-CE82-4BD1-94E5-FC9D257EEC67.jpeg',
'5E735A7F-6CA4-43E6-9F26-6980C3734A28.jpeg',
'B90D35CC-9B78-4EAE-A1C6-7E2F0A351FF5.jpeg',
'1D46240C-F511-4B17-9F25-38DCBE8A5DCA.jpeg',
'FC863DFE-A41F-4A39-85D8-098EE072604B.jpeg',
'0FD71A24-B1F4-48F1-856D-18DC6BEC88F5.jpeg',
'B62A68D0-D5C3-46E2-B19C-0E197901DE98.jpeg',
'7E7CABA5-39BF-42E3-96A4-17A75057E054.jpeg',
'B2F52A50-0FE8-46E7-A074-EAFBED6EA56D.jpeg',
'BC17B8B3-BBC9-4961-A4C6-659D44B6ECC1.jpeg',
'A682CE6B-D80F-4FED-9BF4-728BCD5E3763.jpeg',
'F0D90FBE-F0D2-4EE2-BBBA-1E0883F298E9.jpeg',
'21574B41-AFE7-42EE-8232-937D0591E72B.jpeg',
'9C1D569D-45CD-4E46-BE2F-7AA028B2FC6D.jpeg',
'64458E24-31BB-4F58-8C4D-3D7CB5F1AD5F.jpeg',
'99D07C28-F0CC-4859-A406-35F17F805BDE.jpeg',
'1E703DC2-9051-4084-BD64-51B9DD52F8EB.jpeg',
'011DE72F-C475-4453-B5FC-E7621ABD2EE9.jpeg',
'E30DF9A0-2124-443E-B996-91D31E130ACD.jpeg',
'3ECA2F5D-778E-40B1-9364-D8ED39D0EECE.jpeg',
'4110A9CC-E8F6-4E5F-A6E1-1954E5253889.jpeg',
'A7C0D4BA-3011-4690-A7CB-7B978C4E86BA.jpeg',
'AF8F2B29-FB47-482F-9DDF-95C2383BD995.jpeg',
'489D9CE9-5446-4205-9C88-03ECA9153F74.jpeg',
'BDE3BBB7-2BA1-42A5-B95B-DE998973DD28.jpeg',
'67CBCC78-BEF8-4A21-9519-2DF0E2D60FAC.jpeg',
'510D4432-1779-4458-B00D-AAB852DC7E6F.jpeg',
'E6057E44-568E-4E61-AC7E-7CD7D8C1EC6E.jpeg',
'F71FFDB4-EEFC-4000-9E5F-F3101548EB35.jpeg',
'DA6884BF-ADEA-4021-BDF7-6D33B062D01A.jpeg',
'0C0998A5-2D92-4FD7-9F8D-A436C183F557.jpeg',
'046F8922-8FE9-48FF-895C-0228B6AF04A7.jpeg',
'D289C518-F0BC-44CA-86F3-EAE3E28894EB.jpeg',
'5BAC171C-1E48-49B2-BAE6-09BCCAB46129.jpeg',
'47D7F57F-D2AA-4161-9585-D91B74125EE3.jpeg',
'E7C06C77-0802-42ED-9BD1-7DCD4DC189E6.jpeg',
'EF4062A8-8819-41AA-B981-5CE163F886FA.jpeg',
'875956B3-1F57-40E9-91F7-18C64A67A002.jpeg',
'EEB03181-BB33-4931-8A6A-4FFBC62871D5.jpeg',
'9F38E5AB-283A-4AFB-94BC-BF13E3B8CA4B.jpeg',
'AE49BE88-023C-4826-B2DA-226FDD48D480.jpeg',
'52C6420D-F94A-4C00-8197-FC43929916EE.jpeg',
'41589882-FCD1-4705-9196-EA78694FCEB8.jpeg',
'1EDC5863-50F1-492F-A254-063531F4578B.jpeg',
'FF0B2DC9-3B5A-4246-8953-1FABA6BCCE8E.jpeg',
'6A3F0EFE-7523-4E5A-B9B9-B894BCBDF625.jpeg',
'67BE28F4-6248-47D1-8381-2EAAB5099400.jpeg',
'98E247A5-0DAD-4A1E-A612-2781D54C95BF.jpeg',
'11199B7E-72DD-4A19-86EB-864FC27955C6.jpeg',
'DFE53D18-02E1-490A-A68A-FF2FF0B8A943.jpeg',
'6F519542-E299-4AFC-B425-0FBC7899C9DF.jpeg',
'FFB83BB2-B58B-48C7-9997-962A7D3BC250.jpeg',
'DED11501-1000-41D6-9CB2-C1C8D69CEC58.jpeg',
'C1040660-31B4-486C-9664-700DA51C6A6A.jpeg',
'0FE627F3-2D51-4C74-915B-02B77624B160.jpeg',
'B4B247CE-53F7-45EC-85F3-2BB3D4428F4D.jpeg',
'5CC993A8-7A3B-42D5-B463-BD9289115129.jpeg',
'DB07A5BD-4B8B-46E1-A0FB-C90580D64F43.jpeg',
'AD65F415-989C-4AD7-AD5C-6EBA52A83F2C.jpeg',
'5284A0B6-B603-4674-A782-F56932031DCF.jpeg',
'C3180DD0-A21A-4372-84B8-915B4E00F26F.jpeg',
'C65320A7-6E19-4B3E-8D1E-5A9591EE2864.jpeg',
'EFCD9A23-9EFC-4AEB-9271-C9ADDF62917A.jpeg',
'333F7D58-F6F7-463A-92F6-41D1E97B2DDA.jpeg',
'1B0DD003-6EAF-4B0D-B446-B07984758309.jpeg',
'ED59ED4C-004B-4342-9F79-0C2796764362.jpeg',
'E83878A3-3CFC-42E3-99A3-616193C3BA60.jpeg',
'690E5C47-AA71-45CB-B26D-73020FB04C56.jpeg',
'F4883F0D-7279-4772-B5C3-B3CBDABEFE3D.jpeg',
'795058EC-5A92-474F-8F0A-94DFA74EE4F7.jpeg',
'B3D598C4-9976-482E-B0EC-998F43B518AB.jpeg',
'AA281B12-B71E-49A9-94A3-FD1E20422440.jpeg',
'0AA09892-C260-4BA8-BD5E-7FA0B92596D2.jpeg',
'59B5B52F-9F51-4253-AE1F-27C8E2C7B8ED.jpeg',
'A5693743-A522-44A1-8AF1-EDB9B539150D.jpeg',
'D3DB933C-157F-4852-93A4-CEE6DD4613C8.jpeg',
'661C4A2F-0F58-47ED-958D-4210FCEED34A.jpeg',
'8BA07A9B-9FBB-4298-A4AC-B543E74BFD6D.jpeg',
'783C5FCB-1884-4283-9A53-C2AA49471CB2.jpeg',
'08CB220B-A129-4075-932C-A05D767CBEBE.jpeg',
'3975AE3D-4BD4-4574-8CB1-08678143536B.jpeg',
'82AA9282-A3F3-473C-8893-747ECAED1D04.jpeg',
'12F7CC46-E9A9-4FE8-A356-1DDBB3F032EC.jpeg',
'42BA57F6-F0F0-40B3-8E99-4BC8D1ABF1EA.jpeg',
'D6BB31CC-A165-4DF7-BC72-58C291EADB01.jpeg',
'FD50E18E-FEE1-46BA-BC4C-D83C4D68486A.jpeg',
'928D56DC-5D11-4495-AC6B-5573B33AEE03.jpeg',
'EF224E7F-2986-44AD-9131-4BE6CFA44B73.jpeg',
'143DD69D-81AE-4B76-A9F3-5BBCBB8F6870.jpeg',
'7EB8C457-3D3A-43AE-B953-B268892F28EF.jpeg',
'0A23BF41-28BB-4598-958C-C178CEEBFA73.jpeg',
'21FD9D6F-6B84-45CA-8434-69940BDF99F4.jpeg',
'DF7A1891-6EAF-4499-9BBA-9FD53E3E04B6.jpeg',
'C50A7385-4198-4327-BD7D-08C0091E827A.jpeg',
'EE859ED9-7772-4097-A4E9-086DD29826FD.jpeg',
'058A9737-3326-47E6-ADC7-A65582860539.jpeg',
'923E4A6A-C9BC-4679-968F-498AEACF7DB3.jpeg',
'754E6542-1167-49DA-85C3-1AB4523A41F9.jpeg',
'121891F2-C802-4B5C-B872-38E8AA39F06E.jpeg',
'CE8A6316-D4E4-4AB1-B2A5-D28595B2DAFB.jpeg',
'37E98630-3D54-4941-8640-0CB560FA6102.jpeg']
                   
                   


for image_name in image_name_list:
    img=cv2.imread('bmiapp-data-2017-04-25/'+image_name)
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

bmi_data[['image_name','score_bmi30']].to_csv('img_scores_shiyu.csv')
