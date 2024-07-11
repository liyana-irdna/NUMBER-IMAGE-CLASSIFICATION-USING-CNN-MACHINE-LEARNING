import tkinter as tk
from tkinter import filedialog,Text,Label
import tkinter.font as tkFont
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
new_model = tf.keras.models.load_model('saved_model/my_model')

batch_size = 32
img_height = 300
img_width = 300
class_names = ['Decimal Number','Fraction Number','Negative Number','Percentage Number','Positive Number']
root = tk.Tk()
root.geometry("700x650")
root.title('PROJECT OF NUMBER RECOGNITION')
apps = []

def addApp():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",title="Select File",
    filetypes= (("all files","*.*"),("exe","*.exe")))
    im= Image.open(filename)
    im=im.resize((300,300),Image.ANTIALIAS)
    tkimage=ImageTk.PhotoImage(im)
    myvar=Label(frame,image = tkimage)
    myvar.im = tkimage
    myvar.pack()
    

def runApps():
    global filename
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = tk.Label(frame,text=Output,bg="white")
    labeloutput.pack()

canvas = tk.Canvas(root, height=300, width=300)
canvas.pack()

frame=tk.Frame(root,bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)


fontStyle = tkFont.Font(family="Lucida Grande",size=20)
labeltitle = Label(frame,text="PROJECT OF NUMBER RECOGNITION" ,font = fontStyle,bg="white")
labeltitle.pack()

bg =ImageTk.PhotoImage(Image.open('wallpaper.jpg'))

labelproject = Label(canvas ,image=bg)
labelproject.pack()

openFile = tk.Button(frame,text="Open File",padx=10,pady=10,fg="white", bg="#263D42", command=addApp)
openFile.pack()

runApps=tk.Button(frame,text="Run File",padx=12,pady=10, fg="white", bg="#263D42", command = runApps)
runApps.pack()

root.mainloop()