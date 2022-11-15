import pyttsx3
def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

#facecount collector...
count = 5
label3 = "People Present here is: ",count
c3=3
s3 = "count of people Wearing Mask is: ",c3
c4=2
s4 = "count of people Not Wearing Mask ",c4
s5 = "Please Everyone Wear your mask properly."
#labelvoice = "{0:s}{1:d} {2:s}{3:d}{4:s}{5:d} {6:s}{7:d}{8:s}{9:d} {10:s}".format(label3,count,s3,c3,t1,count,s4,c4,t1,count,s5)
SpeakText(label3)
SpeakText(s3)
SpeakText(s4)
SpeakText(s5)
