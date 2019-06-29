from flask import Flask,request,jsonify
import json
import numpy as np
app=Flask(__name__)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
@app.route('/train')
def train_model():
	df=pd.read_csv('False_ Alarm_Cases.csv')
	df_train=df.iloc[:,1:8]
	print(df_train.shape)
	X=df_train.iloc[:,:6]
	y=df['Spuriosity Index(0/1)']
	model=DecisionTreeClassifier()
	model.fit(X,y)
	joblib.dump(model,'False_alarm_detection.pickle')
	return jsonify({'result':'model has been trained'})
	

@app.route('/test',methods=['POST'])
def test_model():
	clf=joblib.load('False_alarm_detection.pickle')
	request_data=request.get_json()
	type_req_data=type(request_data)
	print(type_req_data)
	a=request_data['Ambient Temperature( deg C)']
	b=request_data['Calibration(days)']
	c=request_data['Unwanted substance deposition(0/1)']
	d=request_data['Humidity(%)']
	e=request_data['H2S Content(ppm)']
	f=request_data['detected by(% of sensors)']
	l=[a,b,c,d,e,f]
	narr=np.array(l)
	narr=narr.reshape(1,6)
	df_test=pd.DataFrame(narr,columns=['Ambient Temperature( deg C)','Calibration(days)','Unwanted substance deposition(0/1)','Humidity(%)','H2S Content(ppm)',
	'detected by(% of sensors)'])
	print(df_test.shape)
	ypred=clf.predict(df_test)
	if ypred==1:
		result="danger"
		
	else:
		result= "safe"
	return jsonify({"recommendation":result})
		

app.run(port=5000)	
	
	