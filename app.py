from flask import Flask, render_template,request, Response,redirect,url_for,session
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import sqlite3
from werkzeug.utils import secure_filename
import face_recognition

app=Flask(__name__)

app.secret_key = "key"
conn=sqlite3.connect('user.db')
# mycursor=conn.cursor()
conn.execute("create table if not exists details(username text unique,name text,email text unique,phoneno text,password text)")
conn.close()

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')


path = 'C:\\Users\\user\\Desktop\\Face Rec\\ImagePage'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 

encodeListKnown = findEncodings(images)
print('Encoding complete - Press Q to close camera')


@app.route('/new',methods = ['POST','GET'])
def new():
    if request.method=='POST':
        name = request.form.get('name')
        print(name)
        

@app.route('/regist')
def register():
    return render_template("regist.html")


@app.route('/details',methods=['POST','GET'])
def details():
    if request.method=='POST':
        cam = cv2.VideoCapture(0)
        name = request.form.get('name')
        fname= request.form.get('fname')
        email= request.form.get('email')
        phone= request.form.get('pno')
        Password= request.form.get('password')
        # cv2.namedWindow("test")

        img_counter = 0
    data=""
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            # pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
            # tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
            directory = r'C:\Users\user\Desktop\Face Rec\ImagePage'
            #  = cv2.imread(path)
            # print(img)
            # os.chdir(directory)
            img_name = ("{}\{}.png".format(directory,name))
            print(os.listdir(directory))
            # cv2.imwrite(img_name, frame)
            saves=cv2.imwrite(img_name,frame)
            print(img_name)
            conn=sqlite3.connect('user.db')
            conn.execute("insert into details values('"+ name +"','"+ fname +"','"+ email +"','"+ phone +"','"+ Password +"')")
            conn.commit()
            conn.close()
            
            if name==img_name:
               
                print("success fully printed")
                
            else:
               
                print("failed ")
    cam.release()

    cv2.destroyAllWindows()
    data = "REGISTERATION FINISHED .."
    return render_template("success.html", data=data)

@app.route('/logs')
def logs():
    return render_template("logs.html")


@app.route('/new_log', methods=['POST','GET'])
def new_log():
    if request.method=='POST':
        names = request.form.get('name')
        print(names)
        print(type(names),"input")
        cap = cv2.VideoCapture(0)
    data="" 
    while True:
        
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        # grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                print(type(name))
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                # if name.lower()=="babin":
                #     print("he is blocked person :")
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                nam = name.lower()
                print(nam)
                if names == nam:
                    # if True:
                    #     data1 = "LOGIN SUCCESS "
                    #     data.append(data1)
                    data = "login success "
                    print("login success ... ")
                    cv2.imshow('Webcam',img)
                    b=cv2.waitKey(1)
                    if b==31 or b==113:
                        print("End Face Detection")
                        break
                    return render_template('success.html',data = data)
                    
                else:
                    print("unknown")
                    data2= "UNKNOWN PERSON "  
                    return render_template('success.html',data = data2)
                    # data.append(data2)
        # return render_template("logs.html")
                    
                    

              
        cv2.imshow('Webcam',img)
        b=cv2.waitKey(1)
        if b==31 or b==113:
            print("End Face Detection")
            break 
        
        return render_template("success.html",data=data)
            # else :
            #     break
@app.route("/login")
def login():
    
    return render_template("login.html")


@app.route("/logins",methods=["POST","GET"])
def logins():
    msg = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form:
        username = request.form['name']
        password = request.form['password']
        conn=sqlite3.connect('user.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM details WHERE username = '"+ username +"' AND password = '"+ password +"'")
        account = cur.fetchone()
        print(account)
        conn.close()
        
        if account:
            session['loggedin'] = True
            session['username'] = account[0]
            msg = "Logged in successfully !"
                
            return render_template('success.html', data = msg)
        else:
            msg = 'Incorrect username / password !'
            return render_template('password.html',data=msg)
    
    return render_template("password.html")




if __name__ == '__main__':
    app.run(debug=True)
