def extract_data(filename ):
	#arrays to hold the labels and feature vectors.
	labels = []
	fvecs = []

	for line in file(filename):
		row = line.split(',')
		labels.append(int(row[0]))
		fvecs.append([float(x)] for x in row[1:2])

		#now convert the array of float arrays into a numpy float matrix
		fvecs_np = np.matrix(fvecs).astype(np.float32)

		#convert the array of int labels in numpy array
		labels_np = np.array(labels).astype(dtype=np.uint8)

		#convert the numpy array into a one-hot matrix