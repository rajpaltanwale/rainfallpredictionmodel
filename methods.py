import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def prediction(sub_index):
	raw_data = pd.read_csv(r'dataset/rainfall_in_india_1901-2015.csv')
	data_cleaned = raw_data.dropna()
	x_names = data_cleaned.SUBDIVISION.unique()
	for i in range(sub_index,sub_index+1):
	    subset_data = data_cleaned[data_cleaned.SUBDIVISION==x_names[i]]
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['ANNUAL','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Annual = required_data['ANNUAL'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Annual, test_size=0.3)
	lr = LinearRegression()
	model = lr.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	pred = pd.DataFrame({'Actual_Annual': y_test.flatten(), 'Predicted_Annual': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['JAN','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jan = required_data['JAN'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jan, test_size=0.3)
	lr = LinearRegression()
	model1 = lr.fit(x_train, y_train)
	y_pred = model1.predict(x_test)
	pred1 = pd.DataFrame({'Actual_Jan': y_test.flatten(), 'Predicted_Jan': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['FEB','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Feb = required_data['FEB'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Feb, test_size=0.3)
	lr = LinearRegression()
	model2 = lr.fit(x_train, y_train)
	y_pred = model2.predict(x_test)
	pred2 = pd.DataFrame({'Actual_Feb': y_test.flatten(), 'Predicted_Feb': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['MAR','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Mar = required_data['MAR'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Mar, test_size=0.3)
	lr = LinearRegression()
	model3 = lr.fit(x_train, y_train)
	y_pred = model3.predict(x_test)
	pred3 = pd.DataFrame({'Actual_Mar': y_test.flatten(), 'Predicted_Mar': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['APR','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Apr = required_data['APR'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Apr, test_size=0.3)
	lr = LinearRegression()
	model4 = lr.fit(x_train, y_train)
	y_pred = model4.predict(x_test)
	pred4 = pd.DataFrame({'Actual_Apr': y_test.flatten(), 'Predicted_Apr': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['MAY','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	May = required_data['MAY'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, May, test_size=0.3)
	lr = LinearRegression()
	model5 = lr.fit(x_train, y_train)
	y_pred = model5.predict(x_test)
	pred5 = pd.DataFrame({'Actual_May': y_test.flatten(), 'Predicted_May': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['JUN','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jun = required_data['JUN'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jun, test_size=0.3)
	lr = LinearRegression()
	model6 = lr.fit(x_train, y_train)
	y_pred = model6.predict(x_test)
	pred6 = pd.DataFrame({'Actual_Jun': y_test.flatten(), 'Predicted_Jun': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['JUL','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jul = required_data['JUL'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jul, test_size=0.3)
	lr = LinearRegression()
	model7 = lr.fit(x_train, y_train)
	y_pred = model7.predict(x_test)
	pred7 = pd.DataFrame({'Actual_Jul': y_test.flatten(), 'Predicted_Jul': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['AUG','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Aug = required_data['AUG'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Aug, test_size=0.3)
	lr = LinearRegression()
	model8 = lr.fit(x_train, y_train)
	y_pred = model8.predict(x_test)
	pred8 = pd.DataFrame({'Actual_Aug': y_test.flatten(), 'Predicted_Aug': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['SEP','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Sep = required_data['SEP'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Sep, test_size=0.3)
	lr = LinearRegression()
	model9 = lr.fit(x_train, y_train)
	y_pred = model9.predict(x_test)
	pred9 = pd.DataFrame({'Actual_Sep': y_test.flatten(), 'Predicted_Sep': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['OCT','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Oct = required_data['OCT'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Oct, test_size=0.3)
	lr = LinearRegression()
	model10 = lr.fit(x_train, y_train)
	y_pred = model10.predict(x_test)
	pred10 = pd.DataFrame({'Actual_Oct': y_test.flatten(), 'Predicted_Oct': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['NOV','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Nov = required_data['NOV'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Nov, test_size=0.3)
	lr = LinearRegression()
	model11 = lr.fit(x_train, y_train)
	y_pred = model11.predict(x_test)
	pred11 = pd.DataFrame({'Actual_Nov': y_test.flatten(), 'Predicted_Nov': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[sub_index]]
	required_data = data_set[['DEC','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Dec = required_data['DEC'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Dec, test_size=0.3)
	lr = LinearRegression()
	model12 = lr.fit(x_train, y_train)
	y_pred = model12.predict(x_test)
	pred12 = pd.DataFrame({'Actual_Dec': y_test.flatten(), 'Predicted_Dec': y_pred.flatten()})
	Bihar = [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred]
	Actual_And_Predicted = pd.concat(Bihar,axis=1)
	x_new = [[2021]]
	z1 = model1.predict(x_new)
	o1 = pd.DataFrame({'Month': 'JAN', 'Rainfall': z1.flatten()})
	z2 = model2.predict(x_new)
	o2 = pd.DataFrame({'Month': 'FEB', 'Rainfall': z2.flatten()})
	z3 = model3.predict(x_new)
	o3 = pd.DataFrame({'Month': 'MAR', 'Rainfall': z3.flatten()})
	z4 = model4.predict(x_new)
	o4 = pd.DataFrame({'Month': 'APR', 'Rainfall': z4.flatten()})
	z5 = model5.predict(x_new)
	o5 = pd.DataFrame({'Month': 'MAY', 'Rainfall': z5.flatten()})
	z6 = model6.predict(x_new)
	o6 = pd.DataFrame({'Month': 'JUN', 'Rainfall': z6.flatten()})
	z7 = model7.predict(x_new)
	o7 = pd.DataFrame({'Month': 'JUL', 'Rainfall': z7.flatten()})
	z8 = model8.predict(x_new)
	o8 = pd.DataFrame({'Month': 'AUG', 'Rainfall': z8.flatten()})
	z9 = model9.predict(x_new)
	o9 = pd.DataFrame({'Month': 'SEP', 'Rainfall': z9.flatten()})
	z10 = model10.predict(x_new)
	o10 = pd.DataFrame({'Month': 'OCT', 'Rainfall': z10.flatten()})
	z11 = model11.predict(x_new)
	o11 = pd.DataFrame({'Month': 'NOV', 'Rainfall': z11.flatten()})
	z12 = model12.predict(x_new)
	o12 = pd.DataFrame({'Month': 'DEC', 'Rainfall': z12.flatten()})
	month_df = [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12]
	Predicted_df = pd.concat(month_df)
	Predicted_df = Predicted_df.round(2)
	z = model.predict(x_new)
	maximum = max(Predicted_df['Rainfall'])
	minimum = min(Predicted_df['Rainfall'])
	aver=sum(Predicted_df['Rainfall'])/len(Predicted_df['Rainfall'])
	average = float("{0:.2f}". format(aver))
	fig = plt.figure(figsize =(7.2, 4.33))
	csfont = {'fontname':'Times New Roman'}

	plt.title("Monthly Rainfall For Year 2022",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)

	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')

	plt.fill_between(Predicted_df['Month'],Predicted_df['Rainfall'],color="orange")
	plt.plot(Predicted_df['Month'],Predicted_df['Rainfall'],color="red")	
	plt.savefig("static/trial.png")	
	
	return Predicted_df, maximum, minimum, average, x_names[sub_index]

def pdata(inp):
	raw_data = pd.read_csv(r'dataset/rainfall_in_india_1901-2015.csv')
	data_cleaned = raw_data.dropna()
	x_names = data_cleaned.SUBDIVISION.unique()
	for i in range(inp,inp+1):
	    subset_data = data_cleaned[data_cleaned.SUBDIVISION==x_names[i]]
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['ANNUAL','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Annual = required_data['ANNUAL'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Annual, test_size=0.3)
	lr = LinearRegression()
	model = lr.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	pred = pd.DataFrame({'Actual_Annual': y_test.flatten(), 'Predicted_Annual': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['JAN','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jan = required_data['JAN'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jan, test_size=0.3)
	lr = LinearRegression()
	model1 = lr.fit(x_train, y_train)
	y_pred = model1.predict(x_test)
	pred1 = pd.DataFrame({'Actual_Jan': y_test.flatten(), 'Predicted_Jan': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['FEB','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Feb = required_data['FEB'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Feb, test_size=0.3)
	lr = LinearRegression()
	model2 = lr.fit(x_train, y_train)
	y_pred = model2.predict(x_test)
	pred2 = pd.DataFrame({'Actual_Feb': y_test.flatten(), 'Predicted_Feb': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['MAR','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Mar = required_data['MAR'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Mar, test_size=0.3)
	lr = LinearRegression()
	model3 = lr.fit(x_train, y_train)
	y_pred = model3.predict(x_test)
	pred3 = pd.DataFrame({'Actual_Mar': y_test.flatten(), 'Predicted_Mar': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['APR','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Apr = required_data['APR'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Apr, test_size=0.3)
	lr = LinearRegression()
	model4 = lr.fit(x_train, y_train)
	y_pred = model4.predict(x_test)
	pred4 = pd.DataFrame({'Actual_Apr': y_test.flatten(), 'Predicted_Apr': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['MAY','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	May = required_data['MAY'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, May, test_size=0.3)
	lr = LinearRegression()
	model5 = lr.fit(x_train, y_train)
	y_pred = model5.predict(x_test)
	pred5 = pd.DataFrame({'Actual_May': y_test.flatten(), 'Predicted_May': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['JUN','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jun = required_data['JUN'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jun, test_size=0.3)
	lr = LinearRegression()
	model6 = lr.fit(x_train, y_train)
	y_pred = model6.predict(x_test)
	pred6 = pd.DataFrame({'Actual_Jun': y_test.flatten(), 'Predicted_Jun': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['JUL','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Jul = required_data['JUL'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Jul, test_size=0.3)
	lr = LinearRegression()
	model7 = lr.fit(x_train, y_train)
	y_pred = model7.predict(x_test)
	pred7 = pd.DataFrame({'Actual_Jul': y_test.flatten(), 'Predicted_Jul': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['AUG','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Aug = required_data['AUG'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Aug, test_size=0.3)
	lr = LinearRegression()
	model8 = lr.fit(x_train, y_train)
	y_pred = model8.predict(x_test)
	pred8 = pd.DataFrame({'Actual_Aug': y_test.flatten(), 'Predicted_Aug': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['SEP','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Sep = required_data['SEP'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Sep, test_size=0.3)
	lr = LinearRegression()
	model9 = lr.fit(x_train, y_train)
	y_pred = model9.predict(x_test)
	pred9 = pd.DataFrame({'Actual_Sep': y_test.flatten(), 'Predicted_Sep': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['OCT','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Oct = required_data['OCT'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Oct, test_size=0.3)
	lr = LinearRegression()
	model10 = lr.fit(x_train, y_train)
	y_pred = model10.predict(x_test)
	pred10 = pd.DataFrame({'Actual_Oct': y_test.flatten(), 'Predicted_Oct': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['NOV','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Nov = required_data['NOV'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Nov, test_size=0.3)
	lr = LinearRegression()
	model11 = lr.fit(x_train, y_train)
	y_pred = model11.predict(x_test)
	pred11 = pd.DataFrame({'Actual_Nov': y_test.flatten(), 'Predicted_Nov': y_pred.flatten()})
	data_set = data_cleaned[data_cleaned.SUBDIVISION == x_names[inp]]
	required_data = data_set[['DEC','YEAR']]
	x = required_data['YEAR'].values.reshape(-1,1)
	Dec = required_data['DEC'].values.reshape(-1,1)
	x_train, x_test, y_train, y_test = train_test_split(x, Dec, test_size=0.3)
	lr = LinearRegression()
	model12 = lr.fit(x_train, y_train)
	y_pred = model12.predict(x_test)
	pred12 = pd.DataFrame({'Actual_Dec': y_test.flatten(), 'Predicted_Dec': y_pred.flatten()})
	Bihar = [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred]
	Actual_And_Predicted = pd.concat(Bihar,axis=1)
	x_new = [[2015]]
	z1 = model1.predict(x_new)
	o1 = pd.DataFrame({'Month': 'JAN', 'Rainfall': z1.flatten()})
	z2 = model2.predict(x_new)
	o2 = pd.DataFrame({'Month': 'FEB', 'Rainfall': z2.flatten()})
	z3 = model3.predict(x_new)
	o3 = pd.DataFrame({'Month': 'MAR', 'Rainfall': z3.flatten()})
	z4 = model4.predict(x_new)
	o4 = pd.DataFrame({'Month': 'APR', 'Rainfall': z4.flatten()})
	z5 = model5.predict(x_new)
	o5 = pd.DataFrame({'Month': 'MAY', 'Rainfall': z5.flatten()})
	z6 = model6.predict(x_new)
	o6 = pd.DataFrame({'Month': 'JUN', 'Rainfall': z6.flatten()})
	z7 = model7.predict(x_new)
	o7 = pd.DataFrame({'Month': 'JUL', 'Rainfall': z7.flatten()})
	z8 = model8.predict(x_new)
	o8 = pd.DataFrame({'Month': 'AUG', 'Rainfall': z8.flatten()})
	z9 = model9.predict(x_new)
	o9 = pd.DataFrame({'Month': 'SEP', 'Rainfall': z9.flatten()})
	z10 = model10.predict(x_new)
	o10 = pd.DataFrame({'Month': 'OCT', 'Rainfall': z10.flatten()})
	z11 = model11.predict(x_new)
	o11 = pd.DataFrame({'Month': 'NOV', 'Rainfall': z11.flatten()})
	z12 = model12.predict(x_new)
	o12 = pd.DataFrame({'Month': 'DEC', 'Rainfall': z12.flatten()})
	month_df = [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12]
	Predicted_df = pd.concat(month_df)
	Predicted_df = Predicted_df.round(2)
	df_for_table = Predicted_df.transpose()
	df_for_table.columns=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'] 
	df_for_table = df_for_table.drop(['Month'], axis=0)
	z = model.predict(x_new)
	maximum = max(Predicted_df['Rainfall'])
	minimum = min(Predicted_df['Rainfall'])
	aver=sum(Predicted_df['Rainfall'])/len(Predicted_df['Rainfall'])
	average = float("{0:.2f}". format(aver))
	fig = plt.figure(figsize =(5.9, 3.8))
	csfont = {'fontname':'Times New Roman'}

	plt.title("Predicted",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)

	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')

	plt.bar(Predicted_df['Month'],Predicted_df['Rainfall'],color="#008080")	
	plt.savefig("static/ppbar.png")	

	fig = plt.figure(figsize =(5.9, 3.8))
	csfont = {'fontname':'Times New Roman'}

	plt.title("Predicted",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)

	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')

	#plt.fill_between(Predicted_df['Month'],Predicted_df['Rainfall'],color="#008080")
	plt.plot(Predicted_df['Month'],Predicted_df['Rainfall'],color="#5D6D7E")	
	plt.savefig("static/ppline.png")

	Rainfall_data = data_cleaned
	ydata = Rainfall_data[(Rainfall_data ['YEAR'] == 2015)]
	subdata = []
	for i in ydata.SUBDIVISION:
	    subdata.append(i)
	final_data = [None] * 36
	for i in range(0,36):
	    final_data[i]=ydata[(ydata['SUBDIVISION'] == subdata[i])]
	npdf = final_data[inp].transpose()
	npdf.columns=['Month']
	xp=npdf[2:14]
	label=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
	fig = plt.figure(figsize =(5.9, 3.8))
	csfont = {'fontname':'Times New Roman'}

	plt.title("Actual",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)

	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')
	plt.bar(label,xp['Month'],color="#008080")
	plt.savefig('static/pbar.png')

	fig = plt.figure(figsize =(5.9, 3.8))
	csfont = {'fontname':'Times New Roman'}
	plt.title("Actual",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)
	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')
	plt.plot(label,xp['Month'],color="#5D6D7E")
	plt.savefig('static/pline.png')

	fig = plt.figure(figsize =(12.5, 4.5))
	csfont = {'fontname':'Times New Roman'}
	plt.title("Actual",color='White',**csfont)
	plt.xlabel("MONTHS",color='White',**csfont)
	plt.ylabel("Rainfall in mm",color='White',**csfont)
	ax = plt.gca()
	ax.set_facecolor('black')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.tick_params(colors='white', which='both')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	fig.patch.set_facecolor('black')
	plt.plot(label,xp['Month'],color="#8B0000")
	plt.fill_between(Predicted_df['Month'],Predicted_df['Rainfall'],color="#228B22")
	plt.plot(Predicted_df['Month'],Predicted_df['Rainfall'],color="#228B22")
	plt.legend(['Actual','Predicted'])
	plt.savefig('static/combined_line.png')
	
	return Predicted_df, final_data[inp], df_for_table