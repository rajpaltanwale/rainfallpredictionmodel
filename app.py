from flask import Flask,render_template, url_for, redirect, flash, request,flash
from forms import *
from methods import *
import pandas as pd
import os

path = os.getcwd()


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = '85c305cad2b32222eadb779ee97871d4'

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/analysis")
def analysis():
    return render_template('Analysis.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/past", methods=['POST','GET'])
def past():
    formp = selectsubdivisionp()
    if request.method == 'POST':
        subdivision_sp = formp.subdp.data
        return redirect(url_for('scriptp', s_divp = subdivision_sp))
    else:
        print("\nIN ELSE\n")
    return render_template('Past.html', title = "Past Prediction", formp = formp)

def sname(val):
    pred_past, datap, df_fr_tbl = pdata(val)
    return render_template('Analysis.html', title = "Past Prediction", fsp = datap, p_past=pred_past, d_tbl=df_fr_tbl)

@app.route("/scriptp<s_divp>")
def scriptp(s_divp):
    sdivp = s_divp

    if sdivp == 'A_N':
        return sname(0)

    elif sdivp == 'A_P':
        return sname(1)

    elif sdivp == 'A_M':
        return sname(2)

    elif sdivp == 'N_M_N_T':
        return sname(3)

    elif sdivp == 'S_H_W_B_S':
        return sname(4)

    elif sdivp == 'G_W_B':
        return sname(5)

    elif sdivp == 'O':
        return sname(6)


    elif sdivp == 'J':
        return sname(7)


    elif sdivp == 'B':
        return sname(8)


    elif sdivp == 'E_U_P':
        return sname(9)


    elif sdivp == 'W_U_P':
        return sname(10)


    elif sdivp == 'U':
        return sname(11)


    elif sdivp == 'H_D_C':
        return sname(12)


    elif sdivp == 'P':
        return sname(13)


    elif sdivp == 'H_P':
        return sname(14)


    elif sdivp == 'J_K':
        return sname(15)


    elif sdivp == 'W_R':
        return sname(16)


    elif sdivp == 'E_R':
        return sname(17)


    elif sdivp == 'W_M_P':
        return sname(18)


    elif sdivp == 'E_M_P':
        return sname(19)


    elif sdivp == 'G_R':
        return sname(20)


    elif sdivp == 'S_K':
        return sname(21)


    elif sdivp == 'K_G':
        return sname(22)


    elif sdivp == 'M_M':
        return sname(23)


    elif sdivp == 'M':
        return sname(24)


    elif sdivp == 'V':
        return sname(25)


    elif sdivp == 'C':
        return sname(26)


    elif sdivp == 'C_A_P':
        return sname(27)


    elif sdivp == 'T':
        return sname(28)


    elif sdivp == 'R':
        return sname(29)


    elif sdivp == 'T_N':
        return sname(30)


    elif sdivp == 'C_K':
        return sname(31)


    elif sdivp == 'N_I_K':
        return sname(32)


    elif sdivp == 'S_I_K':
        return sname(33)


    elif sdivp == 'K':
        return sname(34)


    elif sdivp == 'L':
        return sname(35)


    else:
        print("\nInvalid SubDivision!!\n")
        return render_template('home.html', title = "Home")


@app.route('/future', methods=['POST','GET'])
def future():
    form = selectsubdivision()
    if request.method == 'POST':
        subdivision_s = form.subd.data
        return redirect(url_for('script', s_div = subdivision_s))
    else:
        print("\nIN ELSE\n")
    return render_template('Future.html', title = "Future Prediction", form = form)


def pred(ind):
    sdiv_bihar, maxdivision, mindivision, averagedivision, diviname= prediction(ind)    
    print(maxdivision)
    return render_template('test.html', title = "Result", divi = diviname, fsdiv = sdiv_bihar, maxdiv = maxdivision, mindiv = mindivision, averagediv =averagedivision)    

@app.route("/script<s_div>")
def script(s_div):
    sdiv = s_div

    if sdiv == 'Andaman_Nicobar':
        return pred(0)

    elif sdiv == 'Arunachal_Pradesh':
        return pred(1)

    elif sdiv == 'ASSAM_MEGHALAYA':
        return pred(2)

    elif sdiv == 'Nagalang_Manipur_Nizoram_Tripura':
        return pred(3)

    elif sdiv == 'SUB_HIMALAYAN_WEST_BENGAL_SIKKIM':
        return pred(4)

    elif sdiv == 'GANGETIC_WEST_BENGAL':
        return pred(5)

    elif sdiv == 'ORISSA':
        return pred(6)


    elif sdiv == 'JHARKHAND':
        return pred(7)


    elif sdiv == 'Bihar':
        return pred(8)


    elif sdiv == 'EAST_UTTAR_PRADESH':
        return pred(9)


    elif sdiv == 'WEST_UTTAR_PRADESH':
        return pred(10)


    elif sdiv == 'UTTARAKHAND':
        return pred(11)


    elif sdiv == 'HARYANA_DELHI_CHANDIGARH':
        return pred(12)


    elif sdiv == 'PUNJAB':
        return pred(13)


    elif sdiv == 'HIMACHAL_PRADESH':
        return pred(14)


    elif sdiv == 'JAMMU_KASHMIR':
        return pred(15)


    elif sdiv == 'WEST_RAJASTHAN':
        return pred(16)


    elif sdiv == 'EAST_RAJASTHAN':
        return pred(17)


    elif sdiv == 'WEST_MADHYA_PRADESH':
        return pred(18)


    elif sdiv == 'EAST_MADHYA_PRADESH':
        return pred(19)


    elif sdiv == 'GUJARAT_REGION':
        return pred(20)


    elif sdiv == 'SAURASHTRA_KUTCH':
        return pred(21)


    elif sdiv == 'KONKAN_GOA':
        return pred(22)


    elif sdiv == 'MADHYA_MAHARASHTRA':
        return pred(23)


    elif sdiv == 'MATATHWADA':
        return pred(24)


    elif sdiv == 'VIDARBHA':
        return pred(25)


    elif sdiv == 'CHHATTISGARH':
        return pred(26)


    elif sdiv == 'COASTAL_ANDHRA_PRADESH':
        return pred(27)


    elif sdiv == 'TELANGANA':
        return pred(28)


    elif sdiv == 'RAYALSEEMA':
        return pred(29)


    elif sdiv == 'TAMIL_NADU':
        return pred(30)


    elif sdiv == 'COASTAL_KARNATAKA':
        return pred(31)


    elif sdiv == 'NORTH_INTERIOR_KARNATAKA':
        return pred(32)


    elif sdiv == 'SOUTH_INTERIOR_KARNATAKA':
        return pred(33)


    elif sdiv == 'KERALA':
        return pred(34)


    elif sdiv == 'LAKSHADWEEP':
        return pred(35)


    else:
        print("\nInvalid SubDivision!!\n")
        return render_template('home.html', title = "Home")

if __name__ == '__main__':
    app.run(debug=True)