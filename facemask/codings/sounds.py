import winsound #...[we done]
import time
import cv2
from playsound import playsound

while True:
	playsound('track3.mp3')
	print('playing sound using  playsound')
	time.sleep(5)
	
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
print("Thank You...") 