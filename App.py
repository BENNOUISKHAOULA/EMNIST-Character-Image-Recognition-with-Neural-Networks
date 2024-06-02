
import numpy as np
from tensorflow.keras.optimizers import Adam 
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import cv2

def clear_whiteboard(display):
    wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
    display[wb_y1-10:wb_y2+12, wb_x1-10:wb_x2+12] = (255, 255, 255)


def setup_display():
    title = np.ones((80, 950, 3), dtype=np.uint8) * 255
    title[..., 0] = 237  # Canal bleu
    title[..., 1] = 149  # Canal vert
    title[..., 2] = 100  # Canal rouge

    board = np.ones((490, 650, 3), dtype=np.uint8) * 255

    panel = np.ones((490, 300, 3), dtype=np.uint8) * 255
    panel[..., 0] = 237  # Canal bleu
    panel[..., 1] = 149  # Canal vert
    panel[..., 2] = 100  # Canal rouge

    panel2 = np.ones((100, 950, 3), dtype=np.uint8) * 255
    panel2[..., 0] = 237  # Canal bleu
    panel2[..., 1] = 149  # Canal vert
    panel2[..., 2] = 100  # Canal rouge
    
    board = cv2.rectangle(board, (0, 0), (650, 489), (255, 0, 0), 1)
    panel = cv2.rectangle(panel, (0, 0), (290, 489), (255, 0, 0), 1)
    panel = cv2.rectangle(panel, (22, 310), (268, 480), (255, 0, 0), 1)
    panel = cv2.rectangle(panel, (22, 65), (268, 250), (255, 0, 0), 1)
    
    cv2.line(panel, (145, 311), (145, 479), (255, 0, 0), 1)
    cv2.line(panel, (22, 345), (268, 345), (255, 0, 0), 1)

    cv2.putText(title, "Characters' recognition",(200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.putText(panel, "Action:   ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 0), 1)
    cv2.putText(panel, "The best predictions are:", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 0), 1)
    cv2.putText(panel, "Prediction", (42, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 0), 1)
    cv2.putText(panel, "Accuracy", (168, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 0), 1)
    cv2.putText(panel, actions[0], (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[actions[0]], 1)
    cv2.putText(panel2, "Choose an action: ", (310,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(panel2, "'D'=DRAW ", (130,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(panel2, "'C'=CUT", (310,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    cv2.putText(panel2, "'R'=RESET ", (465,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(panel2, "'E'= EXIT", (670,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    display = np.concatenate((board, panel), axis=1)
    display = np.concatenate((title, display), axis=0)
    display=np.concatenate((display,panel2), axis=0)
    
    return display

def setup_panel(display):
    action_region_pt1, action_region_pt2 = status_regions["action"]
    preview_region_pt1, preview_region_pt2 = status_regions["preview"]
    label_region_pt1, label_region_pt2 = status_regions["labels"]
    acc_region_pt1, acc_region_pt2 = status_regions["accs"]
    
    display[action_region_pt1[1]:action_region_pt2[1], action_region_pt1[0]:action_region_pt2[0]] = (237, 149, 100)
    display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = (237, 149, 100)
    display[label_region_pt1[1]:label_region_pt2[1], label_region_pt1[0]:label_region_pt2[0]] = (237, 149, 100)
    display[acc_region_pt1[1]:acc_region_pt2[1], acc_region_pt1[0]:acc_region_pt2[0]] = (237, 149, 100)
    
    if crop_preview is not None:
        display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = cv2.resize(crop_preview, (crop_preview_h, crop_preview_w)) 
    
    if best_predictions:
        labels = list(best_predictions.keys())
        accs = list(best_predictions.values())
        prediction_status_cordinate = [
            ((725, 460), (825, 460), (0, 0, 255)),
            ((725, 492), (825, 492), (0, 255, 0)),
            ((725, 530), (825, 530), (255, 0, 0))
        ] 
        for i in range(len(labels)):
            label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
            cv2.putText(display, labels[i], label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, str(int(accs[i]*100))+'%', acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        for i in range(len(labels), 3):
            label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
            cv2.putText(display, "_", label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, "_", acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.putText(display, current_action, (745, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[current_action], )


def mouse_click_event(event, x, y, flags, params):
    if current_action is actions[1]:
        whiteboard_draw(event, x, y)
    elif current_action is actions[2]:
        character_crop(event, x, y)


def whiteboard_draw(event, x, y):
    global left_button_down, right_button_down
    
    wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
    if event is cv2.EVENT_LBUTTONUP:
        left_button_down = False
    elif event is cv2.EVENT_RBUTTONUP:
        right_button_down = False
    elif wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
        color = (0, 0, 0)
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_MOUSEMOVE]:
            if event is cv2.EVENT_LBUTTONDOWN:
                color = (0, 0, 0)
                left_button_down = True
            elif left_button_down and event is cv2.EVENT_MOUSEMOVE:
                color = (0, 0, 0)
            elif event is cv2.EVENT_RBUTTONDOWN:
                color = (0, 0, 0)
                right_button_down = True
            elif right_button_down and event is cv2.EVENT_MOUSEMOVE:
                color = (0, 0, 0)
            else:
                return

            cv2.circle(display, (x, y), 10, color, -1)
            cv2.imshow(window_name, display)

def character_crop(event, x, y):

    def arrange_crop_rectangle_cordinates(cor1, cor2):
        if cor1 is None or cor2 is None:
            return
    
        result = ()
        if cor1[1] < cor2[1]:
            if cor1[0] > cor2[0]:
                result = ( (cor2[0], cor1[1]), (cor1[0], cor2[1]) )
            else:
                result = (cor1, cor2)
        else:
            if cor2[0] > cor1[0]:
                result = ( (cor1[0], cor2[1]), (cor2[0], cor1[1]) )
            else:
                result = (cor2, cor1)
        return result

    global bound_rect_cordinates, lbd_cordinate, lbu_cordinate, crop_preview, display, best_predictions
    
    wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
    if wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
        if event is cv2.EVENT_LBUTTONDOWN:
            lbd_cordinate = (x, y)
        elif event is cv2.EVENT_LBUTTONUP:
            lbu_cordinate = (x, y)

        if lbd_cordinate is not None and lbu_cordinate is not None:
            bound_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, lbu_cordinate)
        elif lbd_cordinate is not None:
            if event is cv2.EVENT_MOUSEMOVE:
                mouse_move_cordinate = (x, y)
                mouse_move_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, mouse_move_cordinate)
                top_cordinate, bottom_cordinate = mouse_move_rect_cordinates[0], mouse_move_rect_cordinates[1]
                
                display_copy = display.copy()
                cropped_region = display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]]
                filled_rect = np.zeros((cropped_region.shape[:]))
                filled_rect[:, :, :] = (237, 149, 100)
                filled_rect = filled_rect.astype(np.uint8)
                cropped_rect = cv2.addWeighted(cropped_region, 0.3, filled_rect, 0.5, 1.0)
                
                if cropped_rect is not None:
                    display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]] = cropped_rect
                    cv2.imwrite("captured/filled.jpg", display_copy)
                    cv2.imshow(window_name, display_copy)

        if bound_rect_cordinates is not None:
            top_cordinate, bottom_cordinate = bound_rect_cordinates[0], bound_rect_cordinates[1]
            crop_preview = display[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]].copy()
            crop_preview = np.invert(crop_preview)
            best_predictions = predict(model, crop_preview)
            display_copy = display.copy()
            bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
            setup_panel(display)
            cv2.imshow(window_name, display)
    elif event is cv2.EVENT_LBUTTONUP:
        lbd_cordinate = lbu_cordinate = None
        cv2.imshow(window_name, display)        


def load_model(path):
    model = Sequential()

    model.add(Conv2D(filters=128,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    #model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=512,activation='relu'))
    #model.add(Dropout(.5))
    model.add(Dense(units=47,activation='softmax'))
    #model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights(path)
    
    return model


def predict(model, image):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'd', 'e', 'f', 'g', 'h','n','q', 'r', 't']
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
    prediction = model.predict(image)
    best_predictions = dict()
    
    for i in range(3):
        max_i = np.argmax(prediction[0])
        acc = round(prediction[0][max_i], 1)
        if acc > 0:
            label = labels[max_i]
            best_predictions[label] = acc
            prediction[0][max_i] = 0
        else:
            break
            
    return best_predictions


left_button_down = False
right_button_down = False
bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
whiteboard_region = {"x": (20, 632), "y": (98, 550)}
window_name = "KAPRIL"
best_predictions = dict()
crop_preview_h, crop_preview_w = 243, 182
crop_preview = None
actions = ["NONE", "DRAW", "CUT"]
action_colors = {
    actions[0]: (0, 0, 255),
    actions[1]: (0, 255, 0),
    actions[2]: (0, 255, 192)
}
current_action = actions[0]
status_regions = {
    "action": ((736, 97), (828, 131)),
    "preview": ((674, 147), (917, 329)),
    "labels": ((678, 428), (790, 558)),
    "accs": ((801, 428), (913, 558))
}
#model = load_model("C:/Users/ayoub/Desktop/FINAL CODE/Models/Modeltest1.h5")
model = load_model("./ModelPresentation.h5")


display = setup_display()
cv2.imshow(window_name, display)
cv2.setMouseCallback(window_name, mouse_click_event)
pre_action = None

while True:
    k = cv2.waitKey(1)
    if k == ord('D') or k == ord('C'):
        if k == ord('D'):
            current_action = actions[1]
        elif k == ord('C'):
            current_action = actions[2]
        if pre_action is not current_action:
            setup_panel(display)
            cv2.imshow(window_name, display)
            pre_action = current_action
    elif k == ord('R'):
        clear_whiteboard(display)
        cv2.imshow(window_name, display)
    elif k == ord('E'):
        break
        
cv2.destroyAllWindows()
