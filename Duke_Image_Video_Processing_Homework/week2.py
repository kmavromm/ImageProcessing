import cv2
import numpy as np
import sys

def readImage(directory):
	img = cv2.imread(directory) # does not support raw images
	return img

def readRAW(directory):
	img = open(directory, 'rb')
	return img

def quantizeByCoefficient(block_8x8, N):
	for r in range(8):
		for c in range(8):
			block_8x8[r,c] = (block_8x8[r,c] / N).astype(np.uint8)
			block_8x8[r,c] = block_8x8[r,c] * N
	return block_8x8


def main(inputImage):
	img = readImage(inputImage)
	rows, cols = img.shape[:2]
	i = 0
	jpg = np.zeros((rows,cols), np.float32)
	for r in range(0, rows, 8):
		for c in range(0, cols, 8):
			block_8x8 = img[r:r+8, c:c+8, 0].astype(np.float32)
			dct_8x8 = cv2.dct(block_8x8)
			block_8x8 = quantizeByCoefficient(dct_8x8, 5)
			block_8x8 = np.linalg.inv(dct_8x8) * block_8x8	
		# need to add blocks in jpg
			print((r/8))


if __name__ == '__main__':
	main(sys.argv[1])
