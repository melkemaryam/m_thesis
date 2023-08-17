from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MSE

from preprocessing import Preprocessing
from helper import Helper
from build_sk import Build_sk
from tuning import Tuning
from build_tf import Build_tf

import joblib

class Training():

	def train_sk(self):

		r = Read_data(None, None, None, None, None)
		h = Helper()
		sk = Build_sk()

		# get values
		X_train = r.get_x_train()
		X_test = r.get_x_test()
		y_train = r.get_y_train()
		y_test = r.get_y_test()

		# train the network
		print("[INFO] training network...")
		h.write_report(f"The size of this dataset is %.1f" % (len(X_train) + len(X_test) + len(y_train) + len(y_test)))

		vector, v_train, v_test = sk.get_vector(X_train, X_test)
		model = sk.return_model()

		model.fit(v_train, y_train)
		train_score = model.score(v_train, y_train)

		pred = model.predict(v_test)
		prob = model.predict_proba(v_test)
		test_score = accuracy_score(y_test, pred)
		labels = h.get_labels()

		report = classification_report(y_test, pred, target_names=labels)
		h.write_score(train_score, test_score)
		print(report)
		h.write_report(report)
		h.plot_acc(model, v_train, y_train, 'normal')
		h.plot_loss(model)

		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(args["path"]))
		joblib.dump(model, args["path"])

		return model

	def train_tf(self):

		re = Read_data(None, None, None, None)
		h = Helper()
		b = Build_tf(None)
		t = Tuning(None)
		arg = Args()
		args = arg.parse_arguments()

		MAX_LENGTH = 256

		X_train, X_test, y_train, y_test, tok = re.tokenise(MAX_LENGTH)
		best_hyperparameters = t.get_best_parameters()

		# train the network
		print("[INFO] training network...")
		h.write_report(f"The size of this dataset is %.1f" % (len(X_train) + len(X_test) + len(y_train) + len(y_test)))

		# create logs for Tensorboard
		log_dir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

		# create Early Stopping
		callback = EarlyStopping(monitor='loss', patience=3)

		# get the correct model
		if (args["train"] == 'all' or args["train"] == 'one'):
			tuner = t.return_tuner()
			model = tuner.hypermodel.build(best_hyperparameters)
		elif (args["train"] == 'none' or args["train"] == 'pred'):
			model = b.build_net()

		model.summary()

		# train the model
		history = model.fit(
			validation_data=(X_test, y_test),
			epochs=200,
			batch_size=64, 
			callbacks=[callback, tensorboard],
			verbose=1)

		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(args["path"]))
		model.save(args["path"])

		# evaluate the network
		print("[INFO] evaluating network...")

		# get labels
		labels = h.get_labels()

		# test model
		predictions = model.predict(X_test)
		results = model.evaluate(X_test, y_test)
		report = classification_report(y_test, np.argmax(pred, axis=1), target_names=labels)
		print(report)
		h.write_report(report)

		_, acc = model.evaluate(X_test, test_Y, verbose=0)
		print('> %.3f' % (acc * 100.0))
		h.write_report('> %.3f' % (acc * 100.0))
		h.plot_acc(history, X_train, y_train, op = 'normal')
		print('test_loss:', results[0], 'test_accuracy:', results[1])

		return model


