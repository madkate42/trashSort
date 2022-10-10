import cv2
import numpy as np

import os
from unicodedata import name
from webbrowser import get
import openai
from gtts import gTTS
import os

api_key = "#your_api_key"
openai.organization = "#organization_k"
openai.api_key = api_key

def get_prompt(trash):
    my_prompt = """
    Decide where should trash be put: Household food waste, Recyclable Waste, Hazardous waste, or All other non-recyclable solid waste.

    Trash: Paper
    Bin: Recyclable Waste

    Trash: Mobile Phone
    Bin: Hazardous waste

    Trash: Electronic Device
    Bin: Hazardous waste

    Trash: Fruit peels
    Bin: Household food waste

    Trash: Batteries
    Bin: Hazardous waste

    Trash: Bubble Wrap
    Bin: All other non-recyclable solid waste

    Trash: Clothes
    Bin: Recyclable Waste

    Trash: Any clothing
    Bin: Recyclable waste

    Trash: Dress
    Bin: Recyclable waste

    Trash:
    """
    return my_prompt + trash


def what_bin(trash):
    my_prompt = get_prompt(trash)
    comp = openai.Completion.create(
        engine="text-davinci-002",
        prompt=my_prompt,
        max_tokens=20
    )

    #print(comp)

    #print(type(comp))
    resp = comp["choices"][0]["text"]

    i = (resp.find("Bin:"))
    output = "Use " + resp[(i + 5)::] + " bin."
    print(output)
    return output


def get_advice(trash):
    request = "Give detailed advice on recycling " + trash

    comp = openai.Completion.create(
        engine="text-davinci-002",
        prompt=request,
        max_tokens=200
    )

    output = comp["choices"][0]["text"]
    print(output)
    return output



def description(trash):
    #print("What trash do you have?")
    #trash = input()
    first = what_bin(trash)

    second = get_advice(trash)

    final_text = "If you want to throw away " + trash + " " + first + second
    language = 'en'
    myobj = gTTS(text=final_text, lang=language, slow=False)
    #name_for_the_audio = trash + ".mp3"
    myobj.save("audio_current.mp3")
    #name_for_the_audio = "mpg321 " + name_for_the_audio
    #os.system(name_for_the_audio)
    return


def text_rec():
    print()
    print("WHAT DO YOU NEED TO THROW AWAY?")
    trash = input()
    description(trash)


net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

def video_rec():
    return1 = []
    count = 0
    classes = []


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size = (100,3))
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()
    one_not_fount = True
    while one_not_fount:


        # create a class for each object open cv can detect

        cap = cv2.VideoCapture(0)
        _, img = cap.read()



        # img = cv2.imread('10.jpeg')
        height, width, _ = img.shape


        # resize divide by magic number
        # takes image, normalize,
        # convert image to format for yolo3 no corp
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

        # input processed input in to black box
        net.setInput(blob)

        # detections
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        # how sure we are at prediction
        confidences = []
        class_ids = []

        # first for loop used to extract all the information from extraction
        for output in layerOutputs:
                # second for loop used to extract information for each of the output
            for detection in output:
                # predction starts from the 6th elements
                scores = detection[5:]
                # extract the scores with highest probability
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # if the probability is higher than 70%
                if confidence > 0.2:
                    # rescale picture back to orignal size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    #fetch postion of upper left conner
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)


        # print(len(boxes))
        # normally boxes will overlap since shitty ai
        # prevent overlaping
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.4)
        # print(indexes.flatten())


        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                # print(label)
                # return1.append(label)
                one_not_fount = False
                description(label)
                print("the item is: ", label)
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)


        cv2.imshow('Image',img)
        # key = cv2.waitKey(10000)
        cv2.waitKey(1)
        count = count + 1
        if len(return1) == 100:
            print(return1)
            break


def image_rec():
        return1 = []
        count = 0
        classes = []
        print("text file name: ")
        user_input = input()


        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size = (100,3))
        with open('coco.names', 'r') as f:
            classes = f.read().splitlines()
        one_not_fount = True
        while one_not_fount:
            # create a class for each object open cv can detect

            img = cv2.imread(user_input)
            # _, img = cap.read()
            # print(classes)

            # img = cv2.imread('10.jpeg')
            height, width, _ = img.shape


            # resize divide by magic number
            # takes image, normalize,
            # convert image to format for yolo3 no corp
            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

            # input processed input in to black box
            net.setInput(blob)

            # detections
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            # how sure we are at prediction
            confidences = []
            class_ids = []

            # first for loop used to extract all the information from extraction
            for output in layerOutputs:
                    # second for loop used to extract information for each of the output
                for detection in output:
                    # predction starts from the 6th elements
                    scores = detection[5:]
                    # extract the scores with highest probability
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # if the probability is higher than 70%
                    if confidence > 0.2:
                        # rescale picture back to orignal size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        #fetch postion of upper left conner
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)


            # print(len(boxes))
            # normally boxes will overlap since shitty ai
            # prevent overlaping
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.4)
            # print(indexes.flatten())


            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    # print(label)
                    # return1.append(label)
                    one_not_fount = False
                    description(label)
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)


            cv2.imshow('Image',img)
            # key = cv2.waitKey(10000)
            cv2.waitKey(1)
            count = count + 1
            if len(return1) == 1:
                print(return1)
                break

def main():
    key = 'A'
    while key != 'q' and key != 'Q':
        display_menu()
        key = input()
        if key == 'a' or key == 'A':
            print("pressed")
            video_rec()
            os.system("mpg123 audio_current.mp3")
        if key == 'b' or key == 'B':
            image_rec()
            os.system("mpg123 audio_current.mp3")
        if key == 'c' or key == 'C':
            text_rec()
            os.system("mpg123 audio_current.mp3")

    #ideo_rec()
    #os.system("mpg123 audio_current.mp3")
    return

def display_menu():
    print("WELCOME TO __HAMILTON__COOL__CODE__")
    print("AI ASSISTANT WILL HELP YOU THROW AWAY YOUR TRASH")
    print()
    print("A -- video recognition")
    print("B -- image recognition")
    print("C -- from text")
    print("Q -- to quit")
    print("Choose you option: ")

if __name__ == "__main__":
    main()
