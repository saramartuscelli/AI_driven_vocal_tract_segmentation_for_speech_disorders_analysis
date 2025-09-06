"""
This interface is designed as a tool for clinicians to easily and intuitively interact with the neural network developed in this project.
The Graphic User Interface (GUI) allows visualization of patient dsMRI images as a video and provides automatic segmentation of the articulators in the vocal tract, displaying them to the medical specialist.
Additionally, the interface includes a graphical representation of the articulator area and its variation over the entire protocol duration.
"""

import tkinter as tk
import os
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# DATA IMPORT
path_init = r"C:\Users\saram\PycharmProjects\neuroPW\Dataset_new_new"  # folder where all the patient's images are contained
model = tf.keras.models.load_model('model.keras', compile=False)   # load of the model you want to use to segment

# GLOBAL VARIABLES INITIALIZATION
x_test = []
num_images = 0
y_pred = []
predictions_max = []
region = 0
img_rgba = None
areas = []
time_derivatives = []


def get_patient_images():
    
    """
    Function to load and preprocess patient images. Optionally adds Gaussian noise
    and rescales the image intensities based on the patient's data.
    """

    global x_test
    global num_images
    name = patient_entry.get()  # Get the patient name from the GUI entry field
    img_path = os.path.join(path_init, name)
    data = np.load(img_path)
    x_test = data['images']  # Extract the images (shape: (50, 256, 256, 1))
    num_images = x_test.shape[0]

    # Check if the patient's name contains "s3" -> this is the image of the pathological patient (with no Gaussian Noise)
    if "s3" in name:
        # Add Gaussian noise to the images
        noise = np.random.normal(0, 0.04, x_test.shape)
        x_test = x_test + noise

        # Rescale the image intensities to the range [0.12, 1]
        x_min, x_max = np.min(x_test), np.max(x_test)
        x_test = 0.12 + (x_test - x_min) * (1 - 0.12) / (x_max - x_min)

    return x_test
    

def play_video(tensor, label, delay=100, index=0):

    """
    Function to display a video frame by frame in the GUI.
    """

    if index < tensor.shape[0]:
        # Extract the current frame
        frame = tensor[index, :, :, 0]  # Remove the extra channel

        # Convert the frame to a Tkinter-compatible image
        img = Image.fromarray((frame * 255).astype(np.uint8))  # Scale values from 0-1 to 0-255
        img_tk = ImageTk.PhotoImage(img)

        # Update the label widget with the new image
        label.config(image=img_tk)
        label.image = img_tk

        # Schedule the next frame to be displayed after the specified delay
        label.after(delay, play_video, tensor, label, delay, index + 1)
    else:
        print("End of video")


def on_play_button():
    play_video(x_test, video_label)


def predict():
    
    """
    Function to make predictions on the test data using the trained model.
    The predictions are converted to one-hot encoding for further processing.
    """
    
    global y_pred
    global predictions_max

    # Make predictions using the trained model on the test data
    predictions = model.predict(x_test)

    # Get the class indices with the highest predicted probabilities
    predictions_max = np.argmax(predictions, axis=-1)

    # Convert the predictions to one-hot encoding
    predictions_onehot = tf.keras.utils.to_categorical(predictions_max, num_classes=7)
    y_pred = predictions_onehot

    # Calculate the metrics
    metrics_calculation(num_images)


def metrics_calculation(n):
    
    """
    Function to calculate metrics based on the predictions.
    Computes the area for each class and their time derivatives.
    """
    
    global areas
    global time_derivatives

    areas = np.zeros((n, 7))  # Shape (n, 7), where n is the number of frames and 7 is the number of classes

    for i, pred in enumerate(predictions_max):
        # Calculate the pixel count for each class in the current frame
        areas[i] = np.bincount(pred.flatten(), minlength=7)

    # Calculate the time derivative of areas (difference between consecutive frames)
    time_derivatives = np.diff(areas, axis=0)


def combine_all_frames(index, x, y, color):
    
    """
    Function to overlay the segmentation mask onto the corresponding image.
    """
    
    combined_images = []
    num_frames = x.shape[0]  # Number of frames in the input data

    for i in range(num_frames):
        # Extract the current image frame
        img_frame = x[i, :, :, 0]  # Remove the extra channel
        img = Image.fromarray((img_frame * 255).astype(np.uint8))  # Scale values from 0-1 to 0-255
        img_rgba = img.convert("RGBA")  # Convert the image to RGBA format

        # Extract the current segmentation mask
        mask_frame = y[i, :, :, index]  # Select the mask corresponding to the given index
        mask = Image.fromarray((mask_frame * 255).astype(np.uint8))  # Scale values from 0-1 to 0-255
        mask = mask.convert("L")  # Convert the mask to grayscale for transparency handling

        # Colorize the mask
        mask_coloured = ImageOps.colorize(mask, black="black", white=color)  # Apply the chosen color to the mask
        mask_coloured.putalpha(85)  # Set the transparency level of the mask

        # Combine the image and the colorized mask
        combined = Image.alpha_composite(img_rgba, mask_coloured)

        # Add the combined image to the list
        combined_images.append(combined)

    return combined_images


def play_video_from_images(image_list, label, delay=100, index=0):
    
    """
    Function to display a sequence of images as a video starting from an image list.
    """

    if index < len(image_list):
        # Retrieve the current frame from the image list
        frame = image_list[index]

        # Convert the frame into a Tkinter-compatible image
        img_tk = ImageTk.PhotoImage(frame)

        # Update the label widget with the current frame
        label.config(image=img_tk)
        label.image = img_tk

        # Schedule the next frame to be displayed after the specified delay
        label.after(delay, play_video_from_images, image_list, label, delay, index + 1)
    else:
        print("End of video")



# Functions to display a video where each frame consists of the original image overlaid with the segmentation for each class.
# These functions allow visualizing how the segmentation evolves over time for all the classes.

def show_bkg():
    global region
    region = 0
    combined_frames = combine_all_frames(0, x_test, y_pred, "red")
    play_video_from_images(combined_frames, segmentation_label)

def show_ul():
    global region
    region = 1
    combined_frames = combine_all_frames(1, x_test, y_pred, "yellow")
    play_video_from_images(combined_frames, segmentation_label)

def show_hp():
    global region
    region = 2
    combined_frames = combine_all_frames(2, x_test, y_pred, "orange")
    play_video_from_images(combined_frames, segmentation_label)

def show_sp():
    global region
    region = 3
    combined_frames = combine_all_frames(3, x_test, y_pred, "pink")
    play_video_from_images(combined_frames, segmentation_label)

def show_to():
    global region
    region = 4
    combined_frames = combine_all_frames(4, x_test, y_pred, "purple")
    play_video_from_images(combined_frames, segmentation_label)

def show_ll():
    global region
    region = 5
    combined_frames = combine_all_frames(5, x_test, y_pred, "orange")
    play_video_from_images(combined_frames, segmentation_label)

def show_he():
    global region
    region = 6
    combined_frames = combine_all_frames(6, x_test, y_pred, "brown")
    play_video_from_images(combined_frames, segmentation_label)


def graph(y, graph_title, y_name, y_min, y_max):
    
    """
    Function to plot a graph in a Tkinter window, displaying trends over frames.
    """

    fig = plt.figure(figsize=(7, 4))
    plt.plot(y)
    plt.ylim(y_min, y_max)
    plt.title(graph_title)
    plt.xlabel("Frame")
    plt.ylabel(y_name)

    # Integrate the graph into the Tkinter GUI using a canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()

    # Position the canvas widget in the Tkinter grid
    canvas_widget.grid(row=3, column=4, columnspan=4, padx=5, pady=5)
    canvas.draw()


def show_area():
    title_name = ""
    lim = 0
    if region==0:
        title_name = "Background"
        lim = 2000
    elif region==1:
        title_name = "Upper Lip"
        lim = 50
    elif region == 2:
        title_name = "Hard Palate"
        lim = 50
    elif region == 3:
        title_name = "Soft Palate"
        lim = 50
    elif region==4:
        title_name = "Tongue"
        lim = 200
    elif region==5:
        title_name = "Lower Lip"
        lim = 100
    elif region==6:
        title_name = "Head"
        lim = 800

    graph(areas[:, region], title_name, "Area [pixel]", 0, (np.max(areas[:, region]) + lim))


def show_areavar():
    title_name = ""
    if region==0:
        title_name = "Background"
    elif region==1:
        title_name = "Upper Lip"
    elif region == 2:
        title_name = "Hard Palate"
    elif region == 3:
        title_name = "Soft Palate"
    elif region==4:
        title_name = "Tongue"
    elif region==5:
        title_name = "Lower Lip"
    elif region==6:
        title_name = "Head"

    graph(time_derivatives[:, region], title_name, "Area Variation [pixel/frame]", np.min(time_derivatives[:, region]) - 20, np.max(time_derivatives[:, region]) + 20)


def on_closing():
    print("Closing the application...")
    window.destroy()
    sys.exit(0)


# Create the main window
window = tk.Tk()
window.geometry("1000x600")
window.title("UI")

# Patient label
patient_label = tk.Label(window, text="INSERT PATIENT'S CODE: ")
patient_label.grid(row=0, column=0, sticky="NW", padx= 5, pady=5)

# Create an Entry widget for the patient code
patient_entry = tk.Entry(width=40)
patient_entry.grid(row=0, column=1, columnspan=2, sticky="NW", padx= 5, pady=5)

# Create a button to confirm
patient_button = tk.Button(window, text="Get", command=get_patient_images)
patient_button.grid(row=0, column=3, sticky= "N", padx=10, pady=5)

# Label for the video
video_label = tk.Label(window)
video_label.grid(row=0, column=5, rowspan=2, padx=5)

# Button to play the video
play_button = tk.Button(window, text="Play", command=on_play_button)
play_button.grid(row=0, column=4, sticky= "N", padx=10, pady=5)

# Button to make predictions
predict_button = tk.Button(window, text='SEGMENT', command=predict)
predict_button.grid(row=2, column=5, padx=10, pady=10)

# Label for segmentations
segmentation_label = tk.Label(window)
segmentation_label.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="N")

# Buttons to show the segmentations
frame_seg =tk.Frame()

tk.Label(master=frame_seg, text="SELECT THE REGION TO VISUALIZE:").pack()

BKG_button = tk.Button(master=frame_seg, text='Background', command=show_bkg)
BKG_button.pack(fill=tk.X, pady=1)

UL_button = tk.Button(master=frame_seg, text='Upper Lip', command=show_ul)
UL_button.pack(fill=tk.X, pady=1)

HP_button = tk.Button(master=frame_seg, text='Hard Palate', command=show_hp)
HP_button.pack(fill=tk.X, pady=1)

SP_button = tk.Button(master=frame_seg, text='Soft Palate', command=show_sp)
SP_button.pack(fill=tk.X, pady=1)

TO_button = tk.Button(master=frame_seg, text='Tongue', command=show_to)
TO_button.pack(fill=tk.X, pady=1)

LL_button = tk.Button(master=frame_seg, text='Lower Lip', command=show_ll)
LL_button.pack(fill=tk.X, pady=1)

HE_button = tk.Button(master=frame_seg, text='Head', command=show_he)
HE_button.pack(fill=tk.X, pady=1)

frame_seg.grid(row=3, column=0, sticky="N", padx=10, pady=10)

# Buttons to display the graphs
frame_buttons = tk.Frame()

area_button = tk.Button(master=frame_buttons, text="Area", command=show_area)
area_button.pack(fill=tk.X, padx=10, pady=10)

areavar_button = tk.Button(master=frame_buttons, text="Area Variation", command=show_areavar)
areavar_button.pack(fill=tk.X, padx=10, pady=10)

frame_buttons.grid(row=3, column=3, sticky="N", padx=10, pady=10)

# Bind the window close event to the `on_closing` function
window.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI loop
window.mainloop()



