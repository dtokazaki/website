import numpy as np
import glob
import cv2
import os

def main():
	K=[1,2,3,6,10,20,30]

	trainingFolders=glob.glob(os.path.join(os.path.dirname(__file__),'att_faces_10','training','**','*'))
	
	#convert to matrix
	trainingImages = np.array( [cv2.imread(image) for image in trainingFolders] )

	print("training Images Shape ",trainingImages.shape)

	# read in greyscale
	trainingArray = np.array( np.transpose([np.array(image).flatten() for image in trainingImages[:,:,:,0]] ))

	print("Training Array: " ,trainingArray)

	print("Training Image Shape: ", trainingArray.shape)

	# center matrix 
	meanColumn=np.mean(trainingArray,axis=1)
	meanColumn = np.array(meanColumn).reshape(10304,1)
	
	trainingArray = trainingArray-meanColumn
	
	# run svd
	U, S, Vh = np.linalg.svd(trainingArray, full_matrices=True)
	
	print("U Shape: ", U.shape)

	percentageCorrect = {1:0,2:0,3:0,6:0,10:0,20:0,30:0}
	# for every person
	for num in range(1,11):
		testingFolders=glob.glob(os.path.join(os.path.dirname(__file__),'att_faces_10','testing','s'+str(num),'*'))
			
		# imread
		testingImages = np.array( [np.array(cv2.imread(image)) for image in testingFolders] )

		# flatten
		testingArray = np.array( [np.array(image).reshape(10304,1) for image in testingImages[:,:,:,0]] )
		
		# remove mean Column from testing data
		testingArray = testingArray-meanColumn

		# for every rank
		for rank in K:
			solution = np.transpose(U[:,:rank])
			model = np.dot(solution,trainingArray)
	
			# for every picture
			for test in testingArray:	
				#dot product
				testProjection = np.dot(solution,test)

				#tile
				tile = np.tile(testProjection,(1,60))
					
				dot = model - tile

				# normalize
				dot = np.linalg.norm(dot,axis=0)
				
				# find K closest Neighbor
				minimum = np.argmin(dot,axis=0)
				
				# Check to see if guess is equal to actual
				if (int(minimum/6)+1) == num:
					percentageCorrect[rank]+=1
		testingFolders.clear()

	# Percentage Correct Calculation
	for x in percentageCorrect:
		percentageCorrect[x]=percentageCorrect[x]/40
	print("Percentage Correct (K-Rank,Percentage):" ,percentageCorrect)
	

if __name__ == "__main__":
	main()
	
