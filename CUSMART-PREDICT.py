# importing necessary libraries
import numpy as np
import os.path
from ctypes.wintypes import MAX_PATH
import ctypes
import ctypes.wintypes as w
from pathlib import Path
import shutil
import sys
import pandas as pd
from joblib import load
import shap
from tkinter import *
from tkinter import ttk
from tkinter.font import Font
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import textwrap
from tabulate import tabulate
ctypes.windll.shcore.SetProcessDpiAwareness(0)

# variables to store selected menu values
selected_displayed_value = 0
selected_course_value = 0

# dictionary containing the course names and values
course_items = {33: "Biofuel production technologies", 171: "Animation and multimedia design",
                8014: "Social Service (evening attendance)", 9003: "Agronomy", 9070: "Communication Design",
                9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture", 9147: "Management",
                9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
                9670: "Advertising and Marketing Management", 9773: "Journalism and Communication",
                9853: "Basic Education", 9991: "Management (evening attendance)"}

# get the course dictionary names and keys
courses_names = list(course_items.values())
course_values = list(course_items.keys())

# list containing displayed column values
displaced_items = ["No", "Yes"]

# creates a message dialogue box to alert users
MB_OKCANCEL = 1
IDCANCEL = 2
IDOK = 1

user32 = ctypes.WinDLL('user32')
MessageBox = user32.MessageBoxW
MessageBox.argtypes = w.HWND, w.LPCWSTR, w.LPCWSTR, w.UINT
MessageBox.restype = ctypes.c_int

# Get the documents directory
dll = ctypes.windll.shell32
buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
home_directory = ''
if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
    home_directory = str(buf.value).strip()
else:
    home_directory = str(Path.home()).strip()
    if len(home_directory) == 0 or home_directory == '':
        home_directory = str(os.path.expanduser('~')).strip()

if home_directory[-1] != '\\' or home_directory != '/':
    home_directory = home_directory + '\\'

# Creating directories
home_directory = home_directory + 'Student Performance Prediction\\'

# create the home directory if it does not exist
if not os.path.exists(home_directory):
    os.makedirs(home_directory)

# get the paths for the icon and model files
icon_file = Path(home_directory + "app_logo.ico")
model_file = Path(home_directory + "model.joblib")

# read the model file if it exists
if model_file.is_file():
    model = load(model_file)

# assign a null value for the path variable name
path = ''
prediction = None
probability = None


def update_preview_text(in_text):
    # remove the readonly from text box
    prev_text.configure(state='normal')

    # remove the old data
    prev_text.delete("1.0", "end")

    # add newline and append the data
    prev_text.insert('end', str(in_text))

    # lock back the textbox to readonly
    prev_text.configure(state='disabled')

    # update the UI
    root.update()

    # return back
    return


# function to close window
def close():
    # close the app window
    root.withdraw()
    root.destroy()
    root.quit()


def restart_program():
    # restarts the app when the model file has been moved
    python = sys.executable
    os.execl(python, python, * sys.argv)


def preview_image():
    #################################################
    # create a new popup window for plot preview
    new_window = Toplevel(root)
    new_window.resizable(False, False)
    # Title
    new_window.title("Shap Plot Preview")
    # Geometry
    new_window.geometry('1024x768')

    def show_image(image_id, text_input):
        # check if shap image has been generated
        if not Path(home_directory + f"shap{image_id}.png").is_file():
            # prevent the preview window from showing if the plot images are not found
            new_window.destroy()
            new_window.update()

            return

        # uncomment the code below if you want to display a full preview window
        # new_window.attributes('-fullscreen', True)
        # load the created shap summary image
        image = Image.open(home_directory + f"shap{image_id}.png")

        # resize the Image using resize method and display the image using the tkinter label
        image = image.resize((1024, 650), Image.Resampling.LANCZOS)
        img2 = ImageTk.PhotoImage(image)
        new_label = ttk.Label(new_window, image=img2)
        new_label.grid(row=0, column=0)
        new_label.photo = img2

        # ensure the shap explained text does not exceed the text preview box
        text_input = textwrap.wrap(text_input.strip(), width=120)

        # remove the readonly from text box
        text.configure(state='normal')

        # remove the old data
        text.delete("1.0", "end")

        # add newline and append the data
        text.insert('end', "\n".join(text_input))

        # lock back the textbox to readonly
        text.configure(state='disabled')

    # Create frame to house the text view that describes the shap plot
    textframe = Frame(new_window)
    textframe.grid(in_=new_window, row=1, column=0, padx=5, pady=5, sticky=NSEW)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    # Create preview text box
    text = Text(new_window, width=35, height=2, bg="light cyan", font=text_font)
    text_scrollbar = Scrollbar(new_window)
    text_scrollbar.config(command=text.yview)
    text.config(yscrollcommand=text_scrollbar.set)
    text_scrollbar.pack(in_=textframe, side=RIGHT, fill=Y)
    text.pack(in_=textframe, side=LEFT, fill=BOTH, expand=True)
    # lock back the textbox to readonly
    text.configure(state='disabled')

    # functon to display the shap summary image text
    def first_image():

        # set the text
        text_input = 'The above summary plot unveils the average impact of each feature in a particular class on the model prediction. This offers more insight on which features contributing to either positively or negatively to why the model made decision.'

        # call the function to show image plot and text
        show_image(1, text_input)

        return

    # functon to display the force shap image text
    def second_image():

        # set the text
        text_input = 'The shap force plots for the 3 classes above indicates the impact of each feature on the predictions. those on red indicates they are pushing higher while blue shows lower impacts.'

        # call the function to show image plot and text
        show_image(2, text_input)

        return

    # functon to close shap plot window
    def exit_btn():
        new_window.withdraw()
        new_window.destroy()
        new_window.update()

    # create a frame to house the buttons
    row_1 = Frame(new_window)
    row_1.grid(in_=new_window, row=2, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
    # create the buttons for shap window
    btn1 = Button(row_1, text="View Shap Summary Plot", command=first_image, fg="white", bg="brown", font=('Times', 16))
    btn1.grid(row=2, column=0, padx=10)
    btn2 = Button(row_1, text="View Shap Force Plot", command=second_image, fg="white", bg="brown", font=('Times', 16))
    btn2.grid(row=2, column=1, padx=10)
    btn3 = Button(row_1, text="Close Preview Window", command=exit_btn, fg="white", bg="brown", font=('Times', 16))
    btn3.grid(row=2, column=2, padx=10)

    # display the shap summary image on start
    first_image()


def predict_model():
    global path, model_file, model, prediction, probability

    # get all the input values that will be used for prediction
    val1 = float(selected_course_value)
    val2 = float("0" if ent2.get() == "" else ent2.get())
    val3 = float("0" if ent3.get() == "" else ent3.get())
    val4 = float(selected_displayed_value)
    val5 = float("0" if ent5.get() == "" else ent5.get())
    val6 = float("0" if ent6.get() == "" else ent6.get())
    val7 = float("0" if ent7.get() == "" else ent7.get())
    val8 = float("0" if ent8.get() == "" else ent8.get())
    val9 = float("0" if ent9.get() == "" else ent9.get())
    val10 = float("0" if ent10.get() == "" else ent10.get())

    # create a list using the user input
    new_data = [[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10]]

    # create a list using the input variable names
    col_names = ['Course', 'Previous qualification (grade)', 'Admission grade', 'Displaced',
                 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)',
                 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)',
                 'Curricular units 2nd sem (approved)',	'Curricular units 2nd sem (grade)']

    # create a new pandas dataframe
    df = pd.DataFrame(new_data, columns=[col_names])

    # display table
    print("New DataFrame:")
    print(tabulate(df.iloc[:, [0, 1, 2, 3]], col_names[0:4], tablefmt="grid"))
    print(tabulate(df.iloc[:, [4, 5]], col_names[4:6], tablefmt="grid"))
    print(tabulate(df.iloc[:, [6, 7]], col_names[6:8], tablefmt="grid"))
    print(tabulate(df.iloc[:, [8, 9]], col_names[8:10], tablefmt="grid"))

    # get the input data as an array
    input_data = np.array(new_data)
    print("\nInput Variables: ", ", ".join([str(e) for e in input_data[0]]))

    # perform prediction
    try:
        # perform model prediction using the input data
        prediction = model.predict(input_data)[0]

        # get the prediction probability score value
        probability = model.predict_proba(input_data)[0][prediction]

        # create and save the sharp image
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        # set the font family and font size for the figure text
        font = {'family': 'monospace',
                'weight': 'normal',
                'size': 18}

        plt.rc('font', **font)

        # save shap summary plot images
        fig = plt.figure()
        shap.summary_plot(shap_values, plot_size=[11, 6], feature_names=col_names, plot_type="bar", max_display=25,
                          show=False)
        fig.tight_layout()
        fig.savefig(home_directory + "shap1.png")
        plt.close(fig)

        # save the shap force class 1 image
        shap.force_plot(explainer.expected_value[0], shap_values[0], text_rotation=10, feature_names=col_names,
                        figsize=[17, 4], show=False, matplotlib=True)
        plt.title("Fig 1: SHAP Force plot for Class 0", y=-0.01)
        plt.savefig(home_directory + "shap_1.png")

        # save the shap force class 2 image
        shap.force_plot(explainer.expected_value[1], shap_values[1], text_rotation=10, feature_names=col_names,
                        figsize=[17, 4], show=False, matplotlib=True)
        plt.title("Fig 2: SHAP Force plot for Class 1", y=-0.01)
        plt.savefig(home_directory + "shap_2.png")

        # save the shap force class 3 image
        shap.force_plot(explainer.expected_value[2], shap_values[2], text_rotation=10, feature_names=col_names,
                        figsize=[17, 4], show=False, matplotlib=True)
        plt.title("Fig 3: SHAP Force plot for Class 2", y=-0.01)
        plt.savefig(home_directory + "shap_3.png")

        # combine all the saved images
        image_file_list = [home_directory + "shap_1.png", home_directory + "shap_2.png", home_directory + "shap_3.png"]
        read_images = [Image.open(i) for i in image_file_list]

        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in read_images])[0][1]

        # for a vertical stacking it is simple: use vstack
        combine_images = np.vstack(np.array([i.resize(min_shape) for i in read_images]))
        combine_images = Image.fromarray(combine_images)

        # save the combined shap force image
        combine_images.save(home_directory + 'shap2.png')

        # remove all force shap temporary images
        os.remove(home_directory + "shap_1.png")
        os.remove(home_directory + "shap_2.png")
        os.remove(home_directory + "shap_3.png")

    # capture any possible errors that might arise
    except IOError as e2:
        print("Model prediction error: ", e2)
        # read the model file if it exists
        if model_file.is_file():

            # load the model file
            model = load(model_file)

            # perform model prediction
            prediction = model.predict(input_data)[0]

            # calculate the probability score
            probability = model.predict_proba(input_data)[0][prediction]

    # display model prediction and probability score
    print("Model Prediction: ", prediction)

    # move the scale to the predicted position
    scale.config(state=NORMAL, takefocus=0)
    scale.set(probability * 100)
    scale.config(state=DISABLED, takefocus=0)

    # if prediction is 0
    if prediction == 0:
        # display a brown background color
        scale.configure(background="brown")

        # set the text to the class name Dropout
        developer_label['text'] = "Dropout"

        # set the text background color to brown and text color to white
        developer_label.config(bg="brown", fg="white")

        # display the model's prediction on the text box
        update_preview_text(
            f"Model Prediction: Dropout.\nProbability Score: {round(probability * 100, 2)}%.")
        print(f"Predicted Label: Dropout.\nProbability Score: {round(probability * 100, 2)}%.")

    # if prediction is 1
    elif prediction == 1:
        # display an orange background color
        scale.configure(background="orange")

        # set the text to the class name Enrolled
        developer_label['text'] = "Enrolled"

        # set the text background color to orange and text color to white
        developer_label.config(bg="orange", fg="white")

        # display the model's prediction on the text box
        update_preview_text(
            f"Model Prediction: Enrolled.\nProbability Score: {round(probability * 100, 2)}%.")
        print(f"Predicted Label: Enrolled.\nProbability Score: {round(probability * 100, 2)}%.")

    # if prediction is 2
    elif prediction == 2:

        # display a green background color
        scale.configure(background="green")

        # set the text to the class name Graduate
        developer_label['text'] = "Graduate"

        # set the text background color to green and text color to white
        developer_label.config(bg="green", fg="white")

        # display the model's prediction on the text box
        update_preview_text(
            f"Model Prediction: Graduate.\nProbability Score: {round(probability * 100, 2)}%.")
        print(f"Predicted Label: Graduate.\nProbability Score: {round(probability * 100, 2)}%.")


##
def displayed_func(value):
    global selected_displayed_value
    print(f"Selected Displaced: {ent4.get().strip()}")

    # set selected displaced value as 0 for No and 1 for Yes
    if str(ent4.get()).strip() == "No":

        # set the selected displaced value to 0
        selected_displayed_value = 0

    else:
        # set the selected displaced value to 1
        selected_displayed_value = 1

    return


def course_func(value):
    global selected_course_value
    # searches for the course code using the selected course name
    for code, course in course_items.items():

        # if the selected course name is equal to the current course searched
        if course == str(ent1.get()).strip():

            # stores the new course code to be used for prediction
            selected_course_value = code
            print(f"Selected Course Code: {code}.", f"Selected Course Name: {course}")

    return


# create the tkinter dialogue
root = Tk()
root.geometry("900x550")
root.resizable(False, False)
root.title("Student Performance Prediction")

# create the scale
var = DoubleVar()
scale = Scale(root, variable=var, from_=0, to=100, orient=HORIZONTAL)
scale.grid(in_=root, row=0, column=0, columnspan=2, padx=5, pady=5, sticky=NSEW)
scale.config(state=DISABLED, takefocus=0)
scale.configure(background="gray")

# create the displayed label
developer_label = Label(root, text="No info", font=('Helvetica bold', 28), fg="black")
developer_label.grid(in_=root, row=1, column=0, columnspan=2, rowspan=1, padx=5, pady=5, sticky=NSEW)
developer_label.config(bg="gray", fg="black")

# create a new font
text_font = Font(family="Helvetica", size=14)
############################################################
# create the user input fields
############################################################
##############################################################
# row 1 items
row1 = Frame(root)
row1.grid(in_=root, row=2, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab1 = Label(row1, text='Course: ', font=text_font, anchor='w')
lab1.grid(row=0, column=0, padx=5)
lab1.grid_rowconfigure(0, weight=1)
ent1 = ttk.Combobox(row1, state="readonly", values=courses_names, justify='center', width=29, font=text_font)
# set the first course name as the default value
ent1.set(courses_names[0])
# set the first course code as the new selected course value to use
selected_course_value = course_values[0]
print(f"Selected Course Code: {selected_course_value}.", f"Selected Course Name: {courses_names[0]}.")
ent1.bind("<<ComboboxSelected>>", course_func)
ent1.grid(row=0, column=1, padx=5)
##################################################################################
row2 = Frame(root)
row2.grid(in_=root, row=2, column=1, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab2 = Label(row2, text='Previous qualification (grade): ', font=text_font, anchor='w')
lab2.grid(row=0, column=0, padx=5)
lab2.grid_rowconfigure(0, weight=1)
ent2 = Entry(row2, justify='center', width=10, font=text_font)
ent2.insert(0, "0")
ent2.grid(row=0, column=1, padx=5)
####################################################################################
####################################################################################
# row 2 items
row3 = Frame(root)
row3.grid(in_=root, row=3, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab3 = Label(row3, text='Admission grade: ', font=text_font, anchor='w')
lab3.grid(row=0, column=0, padx=5)
lab3.grid_rowconfigure(0, weight=1)
ent3 = Entry(row3, justify='center', width=10, font=text_font)
ent3.insert(0, "0")
ent3.grid(row=0, column=1, padx=5)

row4 = Frame(root)
row4.grid(in_=root, row=3, column=1, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab4 = Label(row4, text='Displaced: ', font=text_font, anchor='w')
lab4.grid(row=0, column=0, padx=5)
lab4.grid_rowconfigure(0, weight=1)
# create the tkinter menu
ent4 = ttk.Combobox(row4, state="readonly", values=displaced_items, justify='center', width=10, font=text_font)
# set No as the initial displayed item
ent4.set("No")
ent4.bind("<<ComboboxSelected>>", displayed_func)
print(f"Selected Displaced: No")
ent4.grid(row=0, column=1, padx=5)

############################################################
# row 3 items
row5 = Frame(root)
row5.grid(in_=root, row=4, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab5 = Label(row5, text='Curricular units 1st sem (enrolled): ', font=text_font, anchor='w')
lab5.grid(row=0, column=0, padx=5)
lab5.grid_rowconfigure(0, weight=1)
ent5 = Entry(row5, justify='center', width=10, font=text_font)
ent5.insert(0, "0")
ent5.grid(row=0, column=1, padx=5)

row6 = Frame(root)
row6.grid(in_=root, row=4, column=1, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab6 = Label(row6, text='Curricular units 1st sem (approved): ', font=text_font, anchor='w')
lab6.grid(row=0, column=0, padx=5)
lab6.grid_rowconfigure(0, weight=1)
ent6 = Entry(row6, justify='center', width=10, font=text_font)
ent6.insert(0, "0")
ent6.grid(row=0, column=1, padx=5)

############################################################
# row 4 items
row7 = Frame(root)
row7.grid(in_=root, row=5, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab7 = Label(row7, text='Curricular units 1st sem (grade): ', font=text_font, anchor='w')
lab7.grid(row=0, column=0, padx=5)
lab7.grid_rowconfigure(0, weight=1)
ent7 = Entry(row7, justify='center', width=10, font=text_font)
ent7.insert(0, "0")
ent7.grid(row=0, column=1, padx=5)

row8 = Frame(root)
row8.grid(in_=root, row=5, column=1, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab8 = Label(row8, text='Curricular units 2nd sem (enrolled): ', font=text_font, anchor='w')
lab8.grid(row=0, column=0, padx=5)
lab8.grid_rowconfigure(0, weight=1)
ent8 = Entry(row8, justify='center', width=10, font=text_font)
ent8.insert(0, "0")
ent8.grid(row=0, column=1, padx=5)

############################################################
# row 5 items
row9 = Frame(root)
row9.grid(in_=root, row=6, column=0, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab9 = Label(row9, text='Curricular units 2nd sem (approved): ', font=text_font, anchor='w')
lab9.grid(row=0, column=0, padx=5)
lab9.grid_rowconfigure(0, weight=1)
ent9 = Entry(row9, justify='center', width=10, font=text_font)
ent9.insert(0, "0")
ent9.grid(row=0, column=1, padx=5)

row10 = Frame(root)
row10.grid(in_=root, row=6, column=1, columnspan=1, padx=5, pady=5, sticky=NSEW)
lab10 = Label(row10, text='Curricular units 2nd sem (grade): ', font=text_font, anchor='w')
lab10.grid(row=0, column=0, padx=5)
lab10.grid_rowconfigure(0, weight=1)
ent10 = Entry(row10, justify='center', width=10, font=text_font)
ent10.insert(0, "0")
ent10.grid(row=0, column=1, padx=5)

#################################################################################
#################################################################################

# Create frame to house the text box
textframe2 = Frame(root)
textframe2.grid(in_=root, row=9, column=0, columnspan=2, padx=5, pady=5, sticky=NSEW)
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
# Create preview text box
prev_text = Text(root, width=35, height=2, bg="light cyan", font=text_font)
prev_scrollbar = Scrollbar(root)
prev_scrollbar.config(command=prev_text.yview)
prev_text.config(yscrollcommand=prev_scrollbar.set)
prev_scrollbar.pack(in_=textframe2, side=RIGHT, fill=Y)
prev_text.pack(in_=textframe2, side=LEFT, fill=BOTH, expand=True)
prev_text.configure(state='disabled')

# create main window buttons
buttonframe = Frame(root)
buttonframe.grid(in_=root, row=10, column=0, columnspan=2, padx=5, pady=5, sticky=NSEW)
predict_button = Button(buttonframe, text="Predict Student Performance", command=predict_model, fg="white", bg="green", font=text_font)
predict_button.grid(row=0, column=0, padx=10)
predict_button.grid_rowconfigure(0, weight=1)
view_button = Button(buttonframe, text="View Shap Plots", command=preview_image, fg="white", bg="black", font=text_font)
view_button.grid(row=0, column=1, padx=10)
exit_button = Button(buttonframe, text="Close Application", command=close, fg="white", bg="black", font=text_font)
exit_button.grid(row=0, column=2, padx=10)

# close window when escape key is pressed
root.bind("<Escape>", lambda x: root.destroy())


# check for the app icon file
if icon_file.is_file():
    root.iconbitmap(icon_file)

# check if model file is saved
if not model_file.is_file():
    try:
        # copy the model file is it not saved to the documents directory
        shutil.copyfile("model.joblib", model_file)
        # display model file copying progress
        while not model_file.is_file():
            print("Copying model file, please wait!")
            if model_file.is_file():
                # call the function to restart the app after copying the model file
                restart_program()

    # capture any errors if model file fails to copy because is not found in the same directory with the app
    except Exception as e1:
        ret4 = MessageBox(None, f'{e1}\nPlease ensure the model.joblib file is in the same directory with this app.',
                          'Attention User', MB_OKCANCEL)
        # close the app when the user selects the OKAY or CANCEL button
        if ret4 == IDOK:
            close()

        elif ret4 == IDCANCEL:
            close()

# run the application
root.mainloop()
