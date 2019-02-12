# -*- coding: utf-8 -*-

# 建设模型是线性回归，不小心多增加了一列重复特征，使用L1和L2优化有什么不同
import numpy as np
import warnings
from sklearn.exceptions import  ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV, Ridge, Lasso
import matplotlib as mpl
# import matplotlib.pyplot as plt


from sklearn import datasets, cross_validation, discriminant_analysis

def main():
	np.random.seed(22)
	
	N = 9
	# 在0-6之间生成均匀间隔的N个数
	x = np.linspace(0, 6, N) + np.random.randn(N)
	
	print x
	
	print type(np.random.randn(N))
	
	# [-0.09194992 - 0.71335065  2.58179168  2.01067483  2.50887086  2.74772799
	#  5.4188215   4.1463679   6.62649346]
	
	
def linear():
	# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
	np.random.seed(0)
	np.set_printoptions(linewidth=1000)
	N = 2
	x = np.linspace(0, 6, N) + np.random.randn(N)
	x = np.sort(x)
	y = x ** 2 - 4 * x - 3 + np.random.randn(N)
	x.shape = -1, 1
	y.shape = -1, 1
	p = Pipeline([
		('poly', PolynomialFeatures()),
		('linear', LinearRegression(fit_intercept=False))])
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	np.set_printoptions(suppress=True)
	# plt.figure(figsize=(8, 6), facecolor='w')
	d_pool = np.arange(1, N, 1)  # 阶
	m = d_pool.size
	clrs = []  # 颜色
	for c in np.linspace(16711680, 255, m):
		clrs.append('#%06x' % c)
	line_width = np.linspace(5, 2, m)
	# plt.plot(x, y, 'ro', ms=10, zorder=N)
	for i, d in enumerate(d_pool):
		p.set_params(poly__degree=d)
		p.fit(x, y.ravel())
		lin = p.get_params('linear')['linear']
		output = u'%s：%d阶，系数为：' % (u'线性回归', d)
		print output, lin.coef_.ravel()
		x_hat = np.linspace(x.min(), x.max(), num=100)
		x_hat.shape = -1, 1
		y_hat = p.predict(x_hat)
		s = p.score(x, y)
		z = N - 1 if (d == 2) else 0
		label = u'%d阶，$R^2$=%.3f' % (d, s)
	# 	plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
	# 	plt.legend(loc='upper left')
	# 	plt.grid(True)
	# 	# plt.title('线性回归', fontsize=18)
	# 	plt.xlabel('X', fontsize=16)
	# 	plt.ylabel('Y', fontsize=16)
	# plt.show()


def load_data():
	diabetes = datasets.load_diabetes()
	return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


def test_linear_cofficients():
	print("no l1 and l2")
	# X_train, X_test, y_train, y_test = load_data()
	# X_train = np.array([[3, 5], [2, 6], [6, 10]])
	# y_train = np.array([10, 14, 28])
	# Cofficients:[-0.25  3.75], intercept -8.00
	#################################################
	
	# X_train = np.array([[3, 5, 5], [2, 6, 6], [6, 10, 10]])
	# y_train = np.array([10, 14, 28])
	# # Cofficients:[-0.25   1.875  1.875], intercept -8.00
	# #################################################
	
	X_train = np.array([[3, 5, 5 * 2], [2, 6, 6 * 2], [6, 10, 10 * 2]])
	y_train = np.array([10, 14, 28])
	# Cofficients:[-0.25  0.75  1.5 ], intercept -8.00
	#################################################
	
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	
	print("Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))


def test_linear_cofficients_ori():
	print("l1")
	
	X_train = np.array([[3, 5], [2, 6], [6, 10]])
	y_train = np.array([10, 14, 28])
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	print("LinearRegression Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	regr = Lasso()
	regr.fit(X_train, y_train)
	print("Lasso            Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	regr = Ridge()
	regr.fit(X_train, y_train)
	print("Ridge            Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))


	# l1
	# LinearRegression Cofficients:[-0.25  3.75], intercept -8.00
	# Lasso            Cofficients:[ 0.          3.35714286], intercept -6.17
	# Ridge            Cofficients:[ 0.66666667  2.88888889], intercept -5.33


def test_linear_cofficients_one():
	print("l1")
	
	X_train = np.array([[3, 5, 5], [2, 6, 6], [6, 10, 10]])
	y_train = np.array([10, 14, 28])
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	print("LinearRegression Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	regr = Lasso()
	regr.fit(X_train, y_train)
	print("Lasso            Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	regr = Ridge()
	regr.fit(X_train, y_train)
	print("Ridge            Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))


	# l1
	# LinearRegression Cofficients:[-0.25   1.875  1.875], intercept -8.00
	# Lasso            Cofficients:[  0.00000000e+00   3.35714286e+00   3.17206578e-17], intercept -6.17
	# Ridge            Cofficients:[ 0.30705394  1.61825726  1.61825726], intercept -6.45


def test_linear_cofficients_two():
	print("two")
	
	X_train = np.array([[3, 5, 5 * 2], [2, 6, 6 * 2], [6, 10, 10 * 2]])
	y_train = np.array([10, 14, 28])
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	print("LinearRegression  Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	
	regr = Lasso()
	regr.fit(X_train, y_train)
	print("Lasso             Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	
	regr = Ridge()
	regr.fit(X_train, y_train)
	print("Ridge             Cofficients:%s, intercept %.2f" % (regr.coef_, regr.intercept_))
	
	# LinearRegression  Cofficients:[-0.25  0.75  1.5 ], intercept -8.00
	# Lasso             Cofficients:[ 0.          0.          1.73214286], intercept -6.92
	# Ridge             Cofficients:[ 0.04651163  0.69767442  1.39534884], intercept -7.26


if __name__ == "__main__":
	
	# main()
	# linear()
	# test_linear_cofficients_ori()
	# test_linear_cofficients_one()
	test_linear_cofficients_two()

	"""
		1. 对于不带正则优化的线性回归, 截距不变，权重值和特征的关系成线性
		2. l2正则，截距不同，但权重值和特征的关系是线性的
		3. l1正则，解决不同，权重值也不同，重复特征的权重部分可能会收敛至0
	"""




























