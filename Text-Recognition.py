import os
import tkinter
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog

from tkinter.ttk import Combobox
from turtle import textinput

import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from pytesseract import Output
from textblob import TextBlob
from docx import Document


main = tkinter.Tk()
main.title("Automated Text Recognition and Translation System using OpenCV and Tesseract OCR")
main.geometry("1200x1200")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
global filename, file_path,originaltext


def uploadImage():
    global filename, file_path
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(initialdir="TestImages")
    if file_path:
        filename = os.path.basename(file_path)
        messagebox.showinfo("Success", " Image uploaded successfully!")
        text.insert(END, "\n")
        text.insert(END, file_path)
        text.insert(END, ": Image is loaded\n\n")


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def Preprocessing():
    global filename, file_path
    text.delete('1.0', END)
    """### Preprocessing of images using OpenCV
    We will write basic functions for different preprocessing methods
    - grayscaling
    - thresholding
    - dilating
    - eroding
    - opening
    - canny edge detection
    - noise removal
    - deskwing
    - template matching.

    Different methods can come in handy with different kinds of images.
    """

    # Plot original image

    image = cv2.imread(file_path)
    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    plt.imshow(rgb_img)
    plt.title('ORIGINAL IMAGE')
    plt.show()

    # Preprocess image

    gray = get_grayscale(image)
    thresh = thresholding(gray)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(gray, 100, 200)
    images = {'gray': gray,
              'thresh': thresh,
              'opening': opening,
              'canny': canny}

    # Plot images after preprocessing

    fig = plt.figure(figsize=(13, 13))
    ax = []

    rows = 2
    columns = 2
    keys = list(images.keys())
    for i in range(rows * columns):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title('- ' + keys[i])
        plt.imshow(images[keys[i]], cmap='gray')
    # Get OCR output using Pytesseract

    custom_config = r'--oem 3 --psm 6'
    text.insert(END, "\nBasic functions for different preprocessing methods")
    text.insert(END, "\n------------------------------------------")
    text.insert(END, "\nTESSERACT OUTPUT --> ORIGINAL IMAGE IS LOADED.")
    text.insert(END, "\nTESSERACT OUTPUT --> THRESHOLDED IMAGE IS GENERATED.")
    text.insert(END, "\nTESSERACT OUTPUT --> OPENED IMAGE IS GENERATED.")
    text.insert(END, "\nTESSERACT OUTPUT --> CANNY EDGE IMAGE IS GENERATED.")
    text.insert(END, "\n")
    text.insert(END, "Preprocessing of images using OpenCV is completed!!\n\n")


def Boundingbox():
    global filename, file_path
    text.delete('1.0', END)
    """### Bounding box information using Pytesseract

    While running and image through the tesseract OCR engine, pytesseract allows you to get bounding box imformation
    - on a character level
    - on a word level
    - based on a regex template

    We will see how to obtain both
    """

    # Plot character boxes on image using pytesseract.image_to_boxes() function

    image = cv2.imread(file_path)
    h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])

    plt.figure(figsize=(16, 12))
    plt.imshow(rgb_img)
    plt.title('INPUT IMAGE WITH CHARACTER LEVEL BOXES')
    plt.show()

    # Plot word boxes on image using pytesseract.image_to_data() function

    image = cv2.imread(file_path)
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    print('DATA KEYS: \n', d.keys())

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        # condition to only pick boxes with a confidence > 60%
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    plt.figure(figsize=(16, 12))
    plt.imshow(rgb_img)
    plt.title('INPUT IMAGE WITH WORD LEVEL BOXES')
    plt.show()
    text.insert(END, "\n")
    text.insert(END, "Bounding box information using Pytesseract is completed!!\n\n")


def TextRecognition():
    global filename, file_path,originaltext
    text.delete('1.0', END)
    image = cv2.imread(file_path)
    """### Page Segmentation Modes

        There are several ways a page of text can be analysed. The tesseract api provides several page segmentation modes if you want to run OCR on only a small region or in different orientations, etc.

        Here's a list of the supported page segmentation modes by tesseract -

        0    Orientation and script detection (OSD) only.  
        1    Automatic page segmentation with OSD.  
        2    Automatic page segmentation, but no OSD, or OCR.  
        3    Fully automatic page segmentation, but no OSD. (Default)  
        4    Assume a single column of text of variable sizes.  
        5    Assume a single uniform block of vertically aligned text.  
        6    Assume a single uniform block of text.  
        7    Treat the image as a single text line.  
        8    Treat the image as a single word.  
        9    Treat the image as a single word in a circle.  
        10    Treat the image as a single character.  
        11    Sparse text. Find as much text as possible in no particular order.  
        12    Sparse text with OSD.  
        13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.  

        To change your page segmentation mode, change the ```--psm``` argument in your custom config string to any of the above mentioned mode codes.

        ### Detect orientation and script

        You can detect the orientation of text in your image and also the script in which it is written.
        """
    # Output with only english language specified

    custom_config = r'-l eng --oem 3 --psm 6'
    text.insert(END, "Extracted Text is: \n\n")
    text.insert(END, "\n")
    text.insert(END, pytesseract.image_to_string(image, config=custom_config))
    originaltext=pytesseract.image_to_string(image, config=custom_config)
    text.insert(END, "\n")
import moviepy.editor as mp
import speech_recognition as sr
def VideoTextRecognition():
    global file_path,originaltext
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(initialdir="TestVideos")
    print(file_path)
    # Load the video
    video = mp.VideoFileClip(file_path)
    # Extract the audio from the video
    audio_file = video.audio
    audio_file.write_audiofile("audio.wav")
    # Initialize recognizer
    r = sr.Recognizer()
    # Load the audio file
    with sr.AudioFile("audio.wav") as source:
        data = r.record(source)
    originaltext = r.recognize_google(data)
    text.insert(END,"\n")
    text.insert(END,"Extracted Text is:\n")
    text.insert(END, "\n")
    text.insert(END, originaltext)
def AudioTextRecognition():
    global file_path,originaltext
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(initialdir="TestAudio")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        data = r.record(source)
    originaltext = r.recognize_google(data)
    text.insert(END,"\n")
    text.insert(END,"Extracted Text is:\n")
    text.insert(END, "\n")
    text.insert(END, originaltext)

def MultiLangTextRecognition():
    global filename, file_path,originaltext
    text.delete('1.0', END)
    tb = TextBlob(originaltext)
    item = simpledialog.askstring("Language Selection", "Choose the Language to Translate?")
    translated =tb.translate(from_lang='en', to=str(item))
    text.insert(END, "\n")
    text.insert(END, "Original Text is: \n\n")
    text.insert(END, "\n")
    text.insert(END, originaltext)
    text.insert(END, "\n")
    text.insert(END, "\n")
    text.insert(END, "Translated Text is: \n\n")
    text.insert(END, "\n")
    text.insert(END, translated)
    text.insert(END, "\n")


def save_text_file():
    content = text.get("1.0", "end-1c").strip()
    if not content:
        messagebox.showwarning("Warning", "No text to save!")
        return

    file_format = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt"),
                                                          ("Word files", "*.docx")])
    if not file_format:
        return

    if file_format.endswith(".txt"):
        with open(file_format, "w", encoding="utf-8") as f:
            f.write(content)


    elif file_format.endswith(".docx"):
        from docx import Document
        doc = Document()
        doc.add_paragraph(content)
        doc.save(file_format)

    messagebox.showinfo("Success", "Text saved successfully!")


def TextRecognitionWebcam():
    main.destroy()
def close():
    main.destroy()


font = ('times', 14, 'bold')
title = Label(main, text='Automated Text Recognition and Translation System using OpenCV and Tesseract OCR')
title.config(bg='DarkGoldenrod1', fg='black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Test Image", command=uploadImage)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocessing of image using OpenCV", command=Preprocessing)
preprocessButton.place(x=50, y=150)
preprocessButton.config(font=font1)

smoteButton = Button(main, text="Bounding box information using Tesseract", command=Boundingbox)
smoteButton.place(x=50, y=200)
smoteButton.config(font=font1)

existingButton = Button(main, text="Automated Text Recognition using OCR", command=TextRecognition)
existingButton.place(x=50, y=250)
existingButton.config(font=font1)

trButton = Button(main, text="Automated Text Recognition using Video", command=VideoTextRecognition)
trButton.place(x=50, y=300)
trButton.config(font=font1)

tAButton = Button(main, text="Automated Text Recognition using Audio", command=AudioTextRecognition)
tAButton.place(x=50, y=350)
tAButton.config(font=font1)

cnnButton = Button(main, text="Language Translation", command=MultiLangTextRecognition)
cnnButton.place(x=50, y=400)
cnnButton.config(font=font1)

saveButton = Button(main, text="Save Extracted Text", command=save_text_file)
saveButton.place(x=50, y=450)
saveButton.config(font=font1)

dcfButton = Button(main, text="Exit", command=close)
dcfButton.place(x=50, y=500)
dcfButton.config(font=font1)


font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500, y=150)
text.config(font=font1)

main.config(bg='LightSteelBlue1')
main.mainloop()
