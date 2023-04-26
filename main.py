import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Security - Face Recognition")

        # Create a button to start the camera
        self.start_button = tk.Button(master, text="Start Camera", command=self.show_camera)
        self.start_button.pack()

        # Create a button to stop the camera
        self.stop_button = tk.Button(master, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack()

        # Create a button to stop the camera
        self.evaluate_button = tk.Button(master, text="Evaluate", command=self.evaluate)
        self.evaluate_button.pack()

        # Create a status label
        self.label_status = tk.Label(root, text="Status")
        self.label_status.pack()

        # Flag to indicate that is going to evaluate
        self.evaluateFlag = False

        # Create a label to display the camera feed and the identity evaluation results
        self.labelImg = tk.Label(master)
        self.labelImg.pack()

        # Flag to indicate if the camera is running
        self.camera_running = False

    def show_camera(self):
        if self.camera_running:
            return

        self.camera_running = True

        # Open the camera and start capturing frames
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label_status.config(text="Failed to open camera.", fg="red")
            self.camera_running = False
            return

        # Enable the "Stop Camera" button
        self.stop_button.configure(state=tk.NORMAL)

        # Disable the "Start Camera" button
        self.start_button.configure(state=tk.DISABLED)

        # Loop over the frames from the camera
        while self.camera_running:
            # Read a frame from the camera
            ret, frame = self.cap.read()
            if not ret:
                self.label_status.config(text="Failed to read frame from camera.", fg="red")
                self.camera_running = False
                break

            # Resize the frame to a valid size and shape
            if self.evaluateFlag:
                
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.resize(gray_frame, (640, 480))
                    haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    faces = haarcascade.detectMultiScale(gray_frame)
                    for (x, y, w, h) in faces:
                        FaceImg = gray_frame[y:y+h,x:x+w]
                        cv2.imwrite("grayFace.jpg",FaceImg)
                    
                    #cv2.imwrite("test.jpg", gray_frame)

                    # Evaluate the identity of the person in the frame
                    FaceImg = cv2.resize(FaceImg,(224,224))
                    is_joe = self.evaluate_identity(FaceImg)

                    # Update the label with the result
                    if is_joe:
                        self.label_status.config(text="Access Permitted", fg="green")
                    else:
                        self.label_status.config(text="Access Denied", fg="red")
                    
                    self.evaluateFlag = False
            
            # Convert the frame to a format that can be displayed in the GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)

            # Update the GUI
            self.labelImg.configure(image=frame)
            self.labelImg.image = frame
            self.master.update()

            # Wait for 10 milliseconds and check if the user has closed the window or pressed the "Esc" key
            key = cv2.waitKey(10)
            if key == 27:
                self.stop_camera()
                break

        # Release the video capture and destroy the window
        self.stop_camera()

    def stop_camera(self):
        self.camera_running = False
        self.cap.release()
    
    def evaluate(self):
        self.evaluateFlag = True

    def list_content(self,path):
        val = []
        with os.scandir(path) as files:
            for file in files:
                #print(directorio.name)
                val.append(file.name)
        return val


    def evaluate_identity(self, frame):

        path = 'C:/Users/arman/OneDrive/Escritorio/PDS/AutorizedUsers/'
        Users = self.list_content(path)

        for i in range (len (Users)):
            path = 'C:/Users/arman/OneDrive/Escritorio/PDS/AutorizedUsers/' + Users[i]+"/"
            pics = self.list_content(path)
            for j in range (len (pics)):
            # Load the image of Joe and convert it to grayscale
                joe_image = cv2.imread(path+pics[j], cv2.IMREAD_GRAYSCALE)     #Change to the specific user to detect
                joe_image = cv2.resize(joe_image, (224, 224))

                cv2.imwrite("Tomada.jpg",frame)
                cv2.imwrite("Referencia.jpg",joe_image)
                # Apply correlation to find the best match between the Joe image and the input frame        
                result = cv2.matchTemplate(frame, joe_image, cv2.TM_CCOEFF_NORMED)
                max_val = np.max(result)
                print(max_val)

                # Check if the best match is the user
                if max_val > 0.80:
                    print(Users[i],"\n")
                    return True
            
        return False

# Create the GUI object and run the mainloop
root = tk.Tk()
gui = GUI(root)
root.mainloop()
