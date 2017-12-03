import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

############## LOAD DATA ###############

# Load .txt file into list
# -------------------------
# Returns features per vector (fpervec), # data samples (nvecs), 
# all data as list (vecs), list of feature names (features)
def loadFile(x):
	reader = csv.reader(open(x))
	vecs, features = [], []
	fpervec, nvecs, rowcount = 0, 0, 1
	for row in reader:
		if rowcount == 1: 
			fpervec = len(row)
			features = row
		if rowcount > 1: vecs.append(row)
		rowcount += 1
	nvecs = rowcount - 1
	return fpervec, nvecs, vecs, features

# Remove non-quantitative data
# ----------------------------
# Returns updated features per vector (fpervec), modified data as list (vecs)
def keepOnlyQuantData(vecs):
	fpervec = 0
	for i in range(len(vecs)):
		new_vec = []
		for f in vecs[i]:
			if isinstance(f, int) or isinstance(f, float): new_vec.append(f)
		vecs[i] = new_vec
	fpervec = len(vecs[0])
	return fpervec, vecs

# Hardcode input features as particular type
# ------------------------------------------
# Returns modified data as list (vecs)
def modifyTypesInput(vecs):
	for vec in vecs:
		for i in range(len(vec)):
			if i == 2: vec[i] = int(vec[i])
			if i == 5: vec[i] = int(vec[i])
			if i == 10: vec[i] = float(vec[i])
			if i == 15: vec[i] = float(vec[i])
	return vecs

# Hardcode output features as particular type
# -------------------------------------------
# Returns modified data as list (vecs)
def modifyTypesOutput(vecs):
	for vec in vecs:
		for i in range(len(vec)):
			vec[i] = float(vec[i])
	return vecs

# Take all data and return only input samples
# -------------------------------------------
# Returns updated features per vector (fpervec), input data as list (new_vecs)
def getInputVecs(vecs):
	fpervec = 0
	new_vecs = []
	for i in range(len(vecs)):
		new_vec = []
		for j in range(len(vecs[i])):
			if j != 15 and j!= 22 and j!= 23:
				new_vec.append(vecs[i][j])
		new_vecs.append(new_vec)
	fpervec = len(vecs[0])
	return fpervec, new_vecs

# Take all data and return only output samples
# --------------------------------------------
# Returns updated features per vector (fpervec), input data as list (new_vecs)
def getOutputVecs(vecs):
	fpervec = 0
	new_vecs = []
	for i in range(len(vecs)):
		new_vec = []
		for j in range(len(vecs[i])):
			# revenue: j = 15, rating: j = 22, num votes: j = 23
			if j == 22:
				new_vec.append(vecs[i][j])
		new_vecs.append(new_vec)
	fpervec = len(vecs[0])
	return fpervec, new_vecs


############## DATA / FEATURE MODIFICIATION ###############

# Normalize budget feature in data samples
# ----------------------------------------
# Returns updated data as list (vecs)
def normalizeBudget(vecs):
	total = 0
	count = 0
	for vec in vecs:
		if vec[0] != 0:
			total += vec[0]
			count += 1
	avg = total/float(count)
	for vec in vecs: 
		if vec[0] == 0: vec[0] = avg
	return vecs

# Add new feature modeling Budget^2
# ---------------------------------
# Returns updated data as list (vecs)
def addBudgetSquaredF(vecs):
	for vec in vecs:
		squared_budget = vec[0]**2
		vec.append(squared_budget)
	return vecs


############## MODELING / PREDICTION ###############

# Run linear regression given input samples and output samples
# ------------------------------------------------------------
# Prints training and test error
def runLinearRegression(x, y):

	train_size = int(.8*len(x))
	test_size = int(.2*len(x))

	x_train = x[:train_size]
	x_test = x[test_size:]

	y_train = y[:train_size]
	y_test = y[test_size:]

	#regr = linear_model.LinearRegression(normalize = True)
	regr = linear_model.Ridge()
	regr.fit(x_train, y_train)

	y_pred_test = regr.predict(x_test)
	y_pred_train = regr.predict(x_train)

	errorAnalysis(y_pred_test, y_pred_train, y_train, y_test, x_test)

# Print error statistics
# ----------------------
def errorAnalysis(pred_test, pred_train, y_train, y_test, x_test):
	# produce error statistics
	train_error = mean_squared_error(y_train, pred_train)
	test_error = mean_squared_error(y_test, pred_test)
	print "Training Error: ", train_error
	print "Test Error", test_error

	threshold = 1
	numSatisfy = 0
	for i, pred in enumerate(pred_test):
		truth = y_test[i]
		if abs(truth - pred) <= threshold:
			numSatisfy += 1
	print "% Correct Within +/- 1:", numSatisfy/float(len(pred_test))

	# produce plots
	plt.plot(x_test, pred_test, color='blue', linewidth=2)
	plt.ylabel('Rating predictions')
	plt.xlabel('Test input data')
	plt.savefig('plot.png')


################################################################

# Do work
# -------
def main():
	fpervec, nvecs, vecs, features = loadFile('movies_metadata.csv')
	in_fpervec, input_vecs = getInputVecs(vecs)
	out_fpervec, output_vecs = getOutputVecs(vecs)
	mod_input_vecs = modifyTypesInput(input_vecs)
	mod_output_vecs = modifyTypesOutput(output_vecs)
	fpervec, quant_vecs = keepOnlyQuantData(input_vecs)
	mod_quant_vecs = normalizeBudget(quant_vecs)
	mod2_quant_vecs = addBudgetSquaredF(mod_quant_vecs)

	runLinearRegression(quant_vecs, mod_output_vecs)

if __name__ == '__main__':
	main()
