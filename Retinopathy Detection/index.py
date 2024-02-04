from flask import Flask, render_template, request,flash
from flask import Response
from random import randint
import sys
from PIL import Image
import os
import io
import base64
from flask import session

from DR_Detection import detection_img
from werkzeug.utils import secure_filename

from DBconn import DBConnection


app = Flask(__name__)
app.secret_key = "abc"



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/admin_home')
def admin_home():
    return render_template('admin_home.html')


@app.route('/user_home')
def user_home():
    return render_template('user_home.html')


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")



        return ""



@app.route('/user_reg')
def user_reg():
    return render_template('user_reg.html')

@app.route('/user')
def user():
    return render_template('user.html')


@app.route("/userlogin",methods =["GET", "POST"])
def userlogin():
        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from physician where username='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user.html",msg="Invalid Credentials")



        return ""



@app.route("/user_reg2",methods =["GET", "POST"])
def user_reg2():
    try:
        name = request.form.get('name')
        uid = request.form.get('uid')
        pwd = request.form.get('pwd')
        email = request.form.get('email')
        mno = request.form.get('mno')

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from physician where username='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:

            return render_template("user_reg.html", messages="User Id already exists..!")

        else:
            sql = "insert into physician values(%s,%s,%s,%s,%s)"
            values = (name, uid, pwd, email, mno)
            cursor.execute(sql, values)
            database.commit()

        return render_template("user.html",messages="Registered Successfully..! Login Here.")
    except Exception as e:
        print(e)




@app.route('/dr_detection')
def dr_detection():
    return render_template('dr_detection.html')




@app.route("/dr_detection2",methods =["GET", "POST"])
def dr_detection2():
    try:

        image = request.files['file']
        imgdata = secure_filename(image.filename)
        filename=image.filename

        filelist = [ f for f in os.listdir("testimg") ]
        for f in filelist:
            os.remove(os.path.join("testimg", f))

        image.save(os.path.join("testimg", imgdata))

        image_path="../RetinopathyDetection/testimg/"+filename

        result=detection_img(image_path)

        print(type(result))



        uid=session['uid']

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "insert into history values(%s,now(),%s)"
        values = (uid, str(result))
        cursor.execute(sql, values)
        database.commit()




    except Exception as e:
        print(e)

    return render_template("dr_detection_results.html", result=result)

@app.route("/dl_evaluations")
def dl_evaluations():
    try:
        database = DBConnection.getConnection()
        cursor = database.cursor()
        cursor.execute("SELECT *FROM evaluations")
        rows = cursor.fetchall()

    except Exception as e:
        print("Error=" , e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

    return render_template("models_evaluations.html",rawdata=rows)


@app.route("/history")
def history():
    try:
        uid = session['uid']
        database = DBConnection.getConnection()
        cursor = database.cursor()
        cursor.execute("SELECT *FROM history where userid='"+uid+"' ")
        rows = cursor.fetchall()

    except Exception as e:
        print("Error=" , e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

    return render_template("history.html",rawdata=rows)





if __name__ == '__main__':
    app.run(host="localhost", port=3711, debug=True)
