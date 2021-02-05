from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import pickle
import os


app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
# db = SQLAlchemy(app)


def data(pay_amnt, fico, int_rate, term_60, dti):
    train = pd.read_csv(r'C:\Users\sheki\OneDrive\Documents\Data_mining\aman_project\loan_acc.csv', low_memory = True)
    df = train.isnull().mean()*100>30
    train.drop(train.columns[df],axis=1,inplace=True)
    row_clean = train.dropna(axis=0)
    row_clean.replace(to_replace = ["Late (16-30 days)","Late (31-120 days)","In Grace Period"],value = "Late",inplace = True)
    row_clean.replace(to_replace = ["Charged Off"],value = "Default",inplace = True)
    num_data = row_clean.select_dtypes(include=[np.number])
    cat_data = row_clean.select_dtypes(exclude = [np.number])
    vif_func = np.column_stack(num_data)
    vif = [variance_inflation_factor(np.array(num_data), i) for i in range(vif_func.shape[1])]
    vif_information = pd.DataFrame({"Variables":vif_func[0],"VIF_Values":vif})
    column_main = vif_information[vif_information["VIF_Values"] < 10]
    num_main_data = num_data[column_main['Variables']]

    crazy = cat_data.drop(['sub_grade','emp_title','verification_status','title','debt_settlement_flag','hardship_flag','last_pymnt_d','zip_code','loan_status','last_credit_pull_d','earliest_cr_line','issue_d'],axis=1)
    crazy_1 = pd.get_dummies(crazy)

    data_1 = num_main_data.drop(['acc_open_past_24mths','percent_bc_gt_75','num_tl_op_past_12m','tot_coll_amt','inq_last_6mths','num_accts_ever_120_pd','delinq_2yrs','pub_rec','pub_rec_bankruptcies','num_tl_90g_dpd_24m','tax_liens','collections_12_mths_ex_med','chargeoff_within_12_mths','acc_now_delinq','delinq_amnt','num_tl_30dpd','num_tl_120dpd_2m'],axis =1)
    data_2 = crazy_1.drop(['addr_state_AZ','addr_state_MD','addr_state_MA','addr_state_IN','addr_state_TN','addr_state_MN','addr_state_WA','addr_state_MO','grade_F','addr_state_CO','addr_state_NV','addr_state_AL','addr_state_LA','addr_state_WI','purpose_medical','addr_state_SC','purpose_car','addr_state_OK','addr_state_KY','addr_state_KS','addr_state_CT','addr_state_OR','purpose_small_business','addr_state_MS','purpose_vacation','addr_state_NM','purpose_moving','purpose_house','addr_state_UT','addr_state_NH','addr_state_RI','addr_state_NE','addr_state_HI','grade_G','addr_state_WV','addr_state_MT','addr_state_DE','addr_state_AK','addr_state_VT','addr_state_ME','addr_state_SD','addr_state_ND','addr_state_WY','addr_state_DC','addr_state_ID','pymnt_plan_y','purpose_renewable_energy','pymnt_plan_n','home_ownership_ANY','purpose_wedding','home_ownership_NONE','purpose_educational'],axis=1)
    main_data = pd.concat([data_1,data_2],axis = 1)
    X2 = main_data[['last_pymnt_amnt', 'last_fico_range_high', 'int_rate', 'term_ 60 months', 'term_ 36 months', 'dti']]
    Y2 = row_clean['loan_status']

    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.20, random_state=2)
    
    model = RandomForestClassifier()
    model.fit(X2_train,Y2_train)

    with open("model_pickle", "wb") as f:
        pickle.dump(model, f)
   
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'model_pickle')
    model = pickle.load(file_path)
    return model.predict([[pay_amnt, fico, int_rate, dti, term_60]])



# class BlogPost(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(100), nullable = False)
#     content = db.Column(db.Text, nullable = False)
#     author = db.Column(db.String(20), nullable = False, default = 'N/A')
#     date_posted = db.Column(db.DateTime, nullable = False, default = datetime.utcnow)

    # def __repr__(self):
    #     return 'Blog Post ' + str(self.id)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    pay_amnt = request.form['last_pay_amt']
    fico = request.form['fico_range']
    int_rate = request.form['int_rate']
    term = request.form['term']
    dti = request.form['dti']
    if(term=='60'):
        term_60=1
    else:
        term_60=0
    output = data(pay_amnt=pay_amnt, fico=fico, int_rate=int_rate, term_60=term_60, dti=dti)
    # new_post = BlogPost(title=post_title, content=post_content, author=post_author)
    # db.session.add(new_post)
    # db.session.commit()
    return render_template("predict.html", output = output[0])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")

if __name__=='__main__':
    app.run(debug=True)