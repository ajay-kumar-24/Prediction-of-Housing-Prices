from sklearn.datasets import load_boston
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import plotly as py
import pylab
import itertools
import timeit,time
import copy

mean= []
std = []
raw_data = load_boston()
def normalizeTarget(train_data,test_data):
	tempMean = np.mean(train_data,axis = 0)
	tempStd = np.std(train_data,axis = 0)
	data = copy.copy(test_data)
	for i in range(0,len(data)):
		data[i] = (data[i] - tempMean)/tempStd
	return data

def normalize(dataT,mean,std):
	data = np.array(copy.copy(dataT))
	for i in range(0,len(data)):
		for j in range(0,len(data[i])):
			data[i][j] = (data[i][j] - mean[j])/std[j]
	return data

def splitData(data1):
	data = copy.copy(data1)
	test = data[::7]
	train = np.delete(data, np.s_[::7], 0)    
	return [train,test]

def calculateTheta(X1,Y):
	X = np.array(X1)
	X = np.insert(X,0,1,axis=1)
	return np.dot(np.dot(pinv(np.dot(np.transpose(X),X)),(np.transpose(X))),Y)

def plotHistogram(data):
	col = np.shape(data)
	j=0
	for i in range(0,int(col[1])):
		plt.figure("Feature : "+raw_data.feature_names[i])
		plt.hist(data[:,i],bins = 10)
		plt.title("Feature "+str(i+1)+'->'+raw_data.feature_names[i])
		pylab.savefig(str(j)+'.png')
		j+=1

def ridgeThetaCalculate(trainD,trainT,lm):
	X = np.insert(trainD,0,1,axis=1)
	identity = np.identity(X.shape[1])
	identity[0][0] = 0
	return np.dot(pinv(np.dot(np.transpose(X),X)+lm*identity),np.dot((np.transpose(X)),trainT))
	
def meanSquareError(theta,X,Y):
	sqr = 0
	X = np.concatenate(((np.ones((len(X),1))),X),axis =1)
	for i in range(0,len(X)):
		sqr += pow(Y[i]-np.dot(np.transpose(theta),X[i]),2)
	return sqr/len(Y)

def meanError(theta,X,Y):
	sqr = 0
	X = np.concatenate(((np.ones((len(X),1))),X),axis =1)
	for i in range(0,len(X)):
		Y[i]=Y[i]-np.dot(theta.T,X[i])
	return Y

def kFoldSplitData(data,K):
	return np.array_split(data,K)

def calculateRidgeError(trainD,testD,trainT,testT,lm):
	print '!!!---------- Ridge Regression ---------!!!'
	for i in lm:
		theta = ridgeThetaCalculate(trainD,trainT,i)
		print_data('Lambda 		',i)
		print_meanSquareError('train',theta,trainD,trainT)
		print_meanSquareError('test',theta,testD,testT)

def ridgeCrossValidation(dataT,trainT,lm):
	print '!!!---------- Ridge Regression : Cross Validation ---------!!!'
	data = np.array(list(dataT))
	d1 = copy.copy(dataT)
	np.random.seed(0)
	concat = np.insert(d1,0,trainT,axis=1)

	np.random.shuffle(concat)
	tt = concat[:,0]
	data = np.delete(concat,0,1)
	target = kFoldSplitData(tt,10)
	data  = kFoldSplitData(data,10)
	
	for l in lm:
		mse = 0.0
		for i in range(0,len(data)):
			test = data[i]
			testT = target[i]
			train = np.concatenate((np.delete(data,i,0)),axis = 0)
			target_t = np.concatenate(np.delete(target,i,0))
			mean = np.mean(train,axis = 0)
			std = np.std(train,axis = 0)
			train_data = normalize(train,mean,std)
			test_data = normalize(test,mean,std)
			theta = ridgeThetaCalculate(train_data, target_t,l)
			error =  meanSquareError(theta,test_data,testT)
			mse += error
		print_data('Lambda 		',l)
		print_data('meanSquareError for test data',mse/10)

def feature_selection_a(correlation,train_data,test_data,Y1,Y2):
	print '!!!---------- FEATURE SELECTION - A ---------!!!'
	names,theta,data,test = pbruteForceSearch(topCorrelated(correlation,4),train_data,test_data,Y1,raw_data.feature_names)
	print 'Highly correlated features - Top 4 			=	'+','.join(names)
	print_meanSquareError('train',theta,data,Y1)
	print_meanSquareError('test',theta,test,Y2)
	
def print_meanSquareError(st,theta,data,Y):
	print 'meanSquareError for '+st+' data 					=	'+str(meanSquareError(theta,data,Y))

def pbruteForceSearch(i,train_data,test_data,Y1,features):
	data = train_data[:, i]
	test = test_data[:, i]
	theta = calculateTheta(data,Y1)
	names = []
	for r in i:
		names.append(features[r])
	return names,theta,data,test

def bruteForceSearch(train_data,test_data,Y1,Y2):
	print '!!!---------- FEATURE SELECTION - C ---------!!!'
	i = np.arange(13)
	results = itertools.combinations(i,4)
	combi = np.array(list(results))
	j = []
	tot = float('Inf')
	for i in combi:
		data = train_data[:, i]
		test = test_data[:, i]
		theta = calculateTheta(data,Y1)
		if meanSquareError(theta,data,Y1) < tot:
			j = i
			tot = meanSquareError(theta,data,Y1)
	names,theta,data,test=pbruteForceSearch(j,train_data,test_data,Y1,raw_data.feature_names)
	print 'Brute force search - Top 4	 			=	'+','.join(names)
	print_meanSquareError('train',theta,data,Y1)
	print_meanSquareError('test',theta,test,Y2)

def topCorrelated(corr,K):
	absolute = np.absolute(corr)
	return np.argsort(absolute)[::-1][:K]

def residual(train_data,test_data,Yr,Y2,corr):
	top = topCorrelated(corr,1)
	index = []
	index.append(int(top))
	features = raw_data.feature_names
	names,theta,data,test=pbruteForceSearch(top,train_data,test_data,Yr,features)
	features_list = []
	features_list.append((''.join(features[top])))
	Y1 = list(Yr)
	for i in range(3):
		Y = meanError(theta,data,Y1)
		train = np.delete(train_data,top,axis =1)
		features = np.delete(features,top,axis=0)
		cor = correlation(train,Y)
		top = topCorrelated(cor,1)
		index.append(int(top))
		features_list.append(''.join(features[top]))
		data = np.concatenate((data,train[:,top]),axis = 1)
		names,theta,d,test=pbruteForceSearch(index,train_data,test_data,Y,raw_data.feature_names)
		Y1 = Y
	abs_index = []
	for i in features_list:
		abs_index.append(raw_data.feature_names.tolist().index(i))
	return abs_index

def residualFeature(f,train_data,test_data,tar):
	print '!!!---------- FEATURE SELECTION - B ---------!!!'
	t = calculateTheta(train_data[:,f],tar[0])
	lis = []
	for r in f:
		lis.append(raw_data.feature_names[r])
	print 'Highly co-related features\t\t\t\t=\t',','.join(lis)
	print_meanSquareError('train',t,train_data[:,f],tar[0])
	print_meanSquareError('test',t,test_data[:,f],tar[1])

def combinations(train_data):
	i = np.arange(13)
	results = itertools.combinations(i,2)
	list_comb = list(results)
	trainD = train_data
	for i in range(13):
		list_comb.append(tuple((i,i)))
	combi = np.array(list(list_comb))
	for i in combi:
		col = train_data[:,i[0]]*train_data[:,i[1]]
		trainD = np.insert(trainD,-1,col,axis =1)
	return trainD

def find_ind(data):
	indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]
	new_index = [i for i in xrange(np.shape(data)[1]) if i not in indices]
	return data[:,new_index]

def featureExpansion(train_data,test_data,Y1,Y2):
	print '!!!---------- FEATURE EXPANSION  ---------!!!'
	new_train = combinations(train_data)
	new_test = combinations(test_data)
	mean = np.mean(new_train,axis  = 0)
	stad = np.std(new_test,axis =0)
	trd = normalize(new_train,mean,stad)
	ted = normalize(new_test,mean,stad)
	theta = calculateTheta(trd,Y1)
	print_meanSquareError('train',theta,trd,Y1)
	print_meanSquareError('test',theta,ted,Y2)	
	
def print_data(t,data):
	print t+'					=	'+str(data)

def correlation(data,Y):
	col = data.shape[1]
	correlate = []
	for i in range(col):
		X = data[:,i]
		mul = X * Y
		correlate.append((np.mean(mul,axis =0)- (np.mean(X,axis=0)*np.mean(Y,axis=0)))/(np.std(X,axis=0),np.std(Y,axis=0)))
	return np.array(correlate)[:,1]

if __name__=="__main__":	
	np.random.seed(0)
	split_data = splitData(raw_data.data)	
	plotHistogram(split_data[0])
	split_target = splitData(raw_data.target)
	mean = np.mean(split_data[0],axis = 0)
	std = np.std(split_data[0],axis = 0)
	train_data = normalize(split_data[0],mean,std)
	test_data = normalize(split_data[1],mean,std)
	corr= correlation((train_data),(split_target[0]))
	
	print '\n!!!---------- Pearson Co-relation ---------!!!'	
	for i in range(0,len(raw_data.feature_names)):
		print_data(raw_data.feature_names[i],corr[i])
	theta = calculateTheta(train_data,split_target[0])
	print '\n!!!---------- Linear Regression ---------!!!'
	print_meanSquareError('train',(theta),(train_data),(split_target[0]))
	print_meanSquareError('test',(theta),(test_data),(split_target[1]))
	print
	calculateRidgeError((train_data),(test_data),(split_target[0]),(split_target[1]),[0.0001,0.001,0.01,0.1,1,10])
	print
	ridgeCrossValidation((split_data[0]),(split_target[0]),([0.0001,0.001,0.01,0.1,1,10]))
	print 
	feature_selection_a(corr,train_data,test_data,split_target[0],split_target[1])
	print 
	f = residual(train_data,test_data,(split_target[0]),(split_target[1]),corr)
	residualFeature(f,train_data,test_data,split_target)
	print 
	bruteForceSearch(copy.copy(train_data),test_data,(split_target[0]),(split_target[1]))
	print 
	featureExpansion((train_data),(test_data),(split_target[0]),(split_target[1]))
	print