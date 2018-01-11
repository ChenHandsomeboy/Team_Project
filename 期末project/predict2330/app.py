from flask import Flask, render_template, request
from wtforms import Form, TextField, RadioField, validators
import requests
import json
import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble, preprocessing, metrics
import talib

app = Flask(__name__)

######## 爬證交所網站資料
#開高低收
def get_today_price(stock, date):
	url='http://www.twse.com.tw/exchangeReport/STOCK_DAY'
	params = {}
	params['stockNo'] = stock
	params['date'] = date
	params['response'] = json
	res = requests.get(url, params=params)
	s = json.loads(res.text)
	rawData=pd.DataFrame(s['data'],columns=['日期', '成交股數', '成交金額', '開盤價','最高價','最低價','收盤價','漲跌價差','成交筆數'])
	return rawData
#籌碼面
def get_Chips(stocktype, date):
	url = 'http://www.twse.com.tw/fund/T86'
	params = {}
	params['selectType'] = stocktype
	params['date'] = date
	params['response'] = json

	res = requests.get(url, params=params)
	s = json.loads(res.text)
	rawData=pd.DataFrame(s['data'],columns=['證券代號','證券名稱','外陸資買進股數(不含外資自營商)','外陸資賣出股數(不含外資自營商)',
									 '外陸資買賣超股數(不含外資自營商)','外資自營商買進股數','外資自營商賣出股數','外資自營商買賣超股數',
									 '投信買進股數','投信賣出股數','投信買賣超股數','自營商買賣超股數','自營商買進股數(自行買賣)','自營商賣出股數(自行買賣)',
									 '自營商買賣超股數(自行買賣)','自營商買進股數(避險)','自營商賣出股數(避險)','自營商買賣超股數(避險)','三大法人買賣超股數'])
	return rawData

#資券比
def get_lend(stocktype, date):
	url = 'http://www.twse.com.tw/exchangeReport/MI_MARGN'
	params = {}
	params['selectType'] = stocktype
	params['date'] = date
	params['response'] = json

	res = requests.get(url, params=params)
	s = json.loads(res.text)
	rawData=pd.DataFrame(s['data'],columns=s['fields'])
	return rawData
#計算技術指標
def talib2df(talib_output,df):
	if type(talib_output) == list:
		ret = pd.DataFrame(talib_output).transpose()
	else:
		ret = pd.Series(talib_output)
	ret.index = df['close'].index
	return ret
#利用歷史資料建立機器學習模型
def machine(today_X):
	df = pd.read_csv('tsmc.csv',header=None,sep=',')
	#歷史資料技術指標
	df.columns=['date','open','high','low','close','volume','long','short','foreign','invest','self','total']
	tsmc = {
			'close':df.close.dropna().astype(float),
			'open':df.open.dropna().astype(float),
			'high':df.high.dropna().astype(float),
			'low':df.low.dropna().astype(float),
			'volume': df.volume.dropna().astype(float),
			'long': df.long.dropna().astype(float),
			'short': df.short.dropna().astype(float),
			'foreign': df.foreign.dropna().astype(float),
			'invest': df.invest.dropna().astype(float),
			'self': df.self.dropna().astype(float),
			'total': df.total.dropna().astype(float),
		   }
	#計算KD#
	KD = talib2df(talib.abstract.STOCH(tsmc, fastk_period=9),df)
	#計算MACD#
	MACD = talib2df(talib.abstract.MACD(tsmc),df)
	#計算OBV#
	OBV = talib2df(talib.abstract.OBV(tsmc),df)
	#計算威廉指數#
	WILLR = talib2df(talib.abstract.WILLR(tsmc),df)
	#ATR 計算#
	ATR = talib2df(talib.abstract.ATR(tsmc),df)
	#alldata
	tsmc=pd.DataFrame(tsmc)
	tsmc = pd.concat([df,KD,MACD,OBV,WILLR,ATR], axis=1)
	tsmc.columns=['date','open','high','low','close','volume','long','short','foreign','invest','self','total','k','d','dif12','dif26','macd','obv','willr','atr']
	#建立機器學習模型
	# 移除遺漏值、與設漲即為1
	df=tsmc
	df['label']=(df.close - df.close.shift(1)) > 0
	df=df.dropna()

	# 創造 dummy variables
	label_encoder = preprocessing.LabelEncoder()
	encoded_label = label_encoder.fit_transform(df["label"])

	# 建立訓練與測試資料
	tsmc_X = df[['open','high','low','close','volume','long','short','foreign','invest','self','total','k','d','macd','obv','willr','atr']]
	tsmc_y = df['label']
	train_X, test_X, train_y, test_y = model_selection.train_test_split(tsmc_X, tsmc_y, test_size = 0.3)

	# 建立 random forest 模型
	forest = ensemble.RandomForestClassifier(n_estimators = 100)
	forest_fit = forest.fit(train_X, train_y)
	probas = forest.predict_proba(today_X)
	return probas
######## Flask
class UIForm(Form):
	# Ref: https://www.tutorialspoint.com/flask/flask_wtf.htm
	date = TextField('日期:', validators=[validators.required()])

@app.route('/')
def index():   # 首頁
	form = UIForm(request.form)
	return render_template('index.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = UIForm(request.form)
	if  request.method == 'POST' and form.validate():
		date =  int(request.form['date'])
		#將今日資料與之前合併算出技術指標值 預測今日
		#整合資訊2330, 20180105 stock date stocktype, date
		k=get_today_price(2330, date)[-1:]
		k.index=[0]
		#買賣超
		a=get_Chips(24, date)
		a=a[a['證券代號']=='2330']
		a=a.iloc[:,[0,1,4,10,11,18]]
		a.index=[0]
		#資券餘額
		b=get_lend(24, date)
		b=b[b['股票代號']=='2330']
		b=b.iloc[:,[6,12]]
		b.index=[0]

		df1 = pd.concat([k,a,b], axis=1)
		cols = df1.columns.tolist()
		cols=cols[:1]+cols[3:7]+cols[1:2]+cols[15:16]+cols[11:15]
		df1=df1[cols]
		df1.columns=['date','open','high','low','close','volume','long','short','foreign','invest','self','total']
		df1.volume=''.join(df1.volume.values).replace(',','')
		df1.long=''.join(df1.long.values).replace(',','')
		df1.short=''.join(df1.short.values).replace(',','')
		df1.foreign=''.join(df1.foreign.values).replace(',','')
		df1.invest=''.join(df1.invest.values).replace(',','')
		df1.self=''.join(df1.self.values).replace(',','')
		df1.total=''.join(df1.total.values).replace(',','')
		df1[['open','high','low','close','volume','long','short','foreign','invest','self','total']] = df1[['open','high','low','close','volume','long','short','foreign','invest','self','total']].apply(pd.to_numeric)
		df1[['volume','foreign','invest','self','total']] = df1[['volume','foreign','invest','self','total']].apply(lambda x: round(x/1000))
		#合併舊資料做kd
		df = pd.read_csv('tsmc.csv',header=None,sep=',')
		df.columns=['date','open','high','low','close','volume','long','short','foreign','invest','self','total']
		df=df.append(df1)
		tsmc = {
				 'close':df.close.dropna().astype(float),
				 'open':df.open.dropna().astype(float),
				 'high':df.high.dropna().astype(float),
				 'low':df.low.dropna().astype(float),
				 'volume': df.volume.dropna().astype(float),
				 'long': df.long.dropna().astype(float),
				 'short': df.short.dropna().astype(float),
				 'foreign': df.foreign.dropna().astype(float),
				 'invest': df.invest.dropna().astype(float),
				 'self': df.self.dropna().astype(float),
				 'total': df.total.dropna().astype(float),
			   }
		#計算KD#
		KD = talib2df(talib.abstract.STOCH(tsmc, fastk_period=9),df)
		#計算MACD#
		MACD = talib2df(talib.abstract.MACD(tsmc),df)
		#計算OBV#
		OBV = talib2df(talib.abstract.OBV(tsmc),df)
		#計算威廉指數#
		WILLR = talib2df(talib.abstract.WILLR(tsmc),df)
		#ATR 計算#
		ATR = talib2df(talib.abstract.ATR(tsmc),df)
		#alldata
		tsmc=pd.DataFrame(tsmc)
		tsmc = pd.concat([df,KD,MACD,OBV,WILLR,ATR], axis=1)
		tsmc.columns=['date','open','high','low','close','volume','long','short','foreign','invest','self','total','k','d','dif12','dif26','macd','obv','willr','atr']
		tsmc['label']=(tsmc.close - tsmc.close.shift(1)) > 0
		tsmc=tsmc.iloc[-1:]
		#今日資料
		today_X = tsmc[['open','high','low','close','volume','long','short','foreign','invest','self','total','k','d','macd','obv','willr','atr']]
		# 預測
		probas = machine(today_X)

		return render_template( 'results.html',
								open = tsmc['open'][0],
								high = tsmc['high'][0],
								low = tsmc['low'][0],
								close = tsmc['close'][0],
								volume = tsmc['volume'][0],
								long = tsmc['long'][0],
								short = tsmc['short'][0],
								foreign = tsmc['foreign'][0],
								invest = tsmc['invest'][0],
								selff = tsmc['self'][0],
								total = tsmc['total'][0],
								k = round(tsmc['k'][0],2),
								d = round(tsmc['d'][0],2),
								macd = round(tsmc['macd'][0],2),
								obv = round(tsmc['obv'][0],2),
								willr = round(tsmc['willr'][0],2),
								atr = round(tsmc['atr'][0],2),
								value = round(probas[0][1], 2))
								
	return render_template('index.html', form=form)

if __name__ == '__main__':
	app.run(debug=True)
