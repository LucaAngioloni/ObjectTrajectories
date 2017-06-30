import argparse
import colorsys
import imghdr
import os
import random
import logging
import math
import datetime
import pickle

from geopy.distance import vincenty

from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head

saveImages = True #True if you wat to output elaborated images to Output folder


# Call
# python3 YOLOVideo.py -i /Users/LucaAngioloni/Desktop/Progetto\ Image\ Analysis/Dataset/Dataset\ 3\ \(Bici\)/2011_09_26\ 3/2011_09_26_drive_0005_sync/image_02/data/ -o /Users/LucaAngioloni/Desktop/Out

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model',
    default='model_data/yolo.h5')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-i',
    '--input_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')
parser.add_argument(
    '-ox',
    '--oxts',
    help='path to oxts files folder',
    default='../../oxts')
parser.add_argument(
    '-ts',
    '--time_stamps',
    help='path to time stamps file',
    default='../timestamps.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

 # image coordinates
 # -------------------> x
 # |
 # |
 # |
 # |
 # |
 # |
 # v
 #
 # y

yaw = np.array([])
position_diff_x = []
position_diff_y = []

data_frames = []

def parse_oxts(oxts_path, timestamps):
    ts = [] #vector containing frames timestamps
    oxts_ts = [] #vector containing oxts timestamps
    
    y = [] #vector containing yaw values at oxts timestamps
    p_lat = [] #vector containing position lat values at oxts timestamps
    p_lng = [] #vector containing position lat values at oxts timestamps
    
    f = open(timestamps, 'r')
    next = f.readline()
    while next != "":
        next = next[:-4]
        tupleTime = datetime.datetime.strptime(next, "%Y-%m-%d %H:%M:%S.%f")
        microsecond = tupleTime.timestamp()
        ts.append(microsecond)
        next = f.readline()
    f.close()
    
    oxts_timestamps = oxts_path + "/timestamps.txt"
    f = open(oxts_timestamps, 'r')
    next = f.readline()
    while next != "":
        next = next[:-4]
        tupleTime = datetime.datetime.strptime(next, "%Y-%m-%d %H:%M:%S.%f")
        microsecond = tupleTime.timestamp()
        oxts_ts.append(microsecond)
        next = f.readline()
    f.close()
    
    ts = np.array(ts)
    oxts_ts = np.array(oxts_ts)
    
    oxts_data = oxts_path + "/data"
    for data_file in os.listdir(oxts_data):
        f = open(oxts_data + "/" + data_file, 'r')
        data = f.read()
        values = data.split()
        f.close()
        
        p_lat.append(float(values[0]))
        p_lng.append(float(values[1]))
        y.append(float(values[5]))
        
    p_lat = np.array(p_lat)
    p_lng = np.array(p_lng)
    y = np.array(y)
    
    #Interpolations
    global yaw
    yaw = np.interp(ts, oxts_ts, y) #positive counter-clockwise, 0 is east

    interp_lat = np.interp(ts, oxts_ts, p_lat)
    interp_lng = np.interp(ts, oxts_ts, p_lng)

    #convert lat and lng to meters
    global position_diff_x
    global position_diff_y
    for i in range(interp_lat.size):
        position_diff_x.append(vincenty((interp_lat[0], interp_lng[0]), (interp_lat[0], interp_lng[i])).meters * np.sign(interp_lng[i] - interp_lng[0])) #longitude is X ------->
        position_diff_y.append(vincenty((interp_lat[0], interp_lng[0]), (interp_lat[i], interp_lng[0])).meters * np.sign(interp_lat[i] - interp_lat[0])) #latitude is Y ^


camera_viewAngleX = 90/180 * math.pi #radians
camera_viewAngleY = 35/180 * math.pi #radians
pixel_width = 1242.0 #pixels
pixel_height = 375.0 #pixels

focal_length = 0.004 # meters, but website sais 4-8 mm
pixel_dim = 0.00000465 # meters (4.65e-6)
camera_height = 1.65 # meters


def computeCoordinates(bounding_box, frame_idx):
    #calculate x,y using plane information and bounding box
    top_left_y, top_left_x, bottom_right_y, bottom_right_x = bounding_box

    # box_x and y are the coordinates of bottom center of the box (touching the ground)
    box_x = (top_left_x + bottom_right_x)/2
    box_y = bottom_right_y

    # print("x1 :" + str(top_left_x) + " y1 :" + str(top_left_y) + " x2 :" + str(bottom_right_x) +" y2 :" + str(bottom_right_y))

    if box_y < (pixel_height/2) + 1:
        return None

    disp_x = box_x - pixel_width/2
    disp_y = box_y - pixel_height/2 #always positive

    # project into plane with focal lenght
    # y = (focal_length/(disp_y * pixel_dim)) * camera_height #always positive
    # x = ((disp_x * pixel_dim)/focal_length) * y 

    # project into plane with view angles
    y = np.tan((math.pi/2 - ((box_y - pixel_height/2)/pixel_height) * camera_viewAngleY)) * camera_height
    x = np.tan(((box_x - pixel_width/2)/pixel_width) * camera_viewAngleX) * y
    
    car_pos = get_Car_Pos_at_Frame(frame_idx)
    dx = car_pos[0]
    dy = car_pos[1]
    
    #Radians
    theta = get_Car_Direction_at_Frame(frame_idx)
    starting_theta = get_Car_Direction_at_Frame(0)
    
    d_theta = theta - starting_theta
    
    T = np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) # translation matrix using dx, dy
    R = np.matrix([[np.cos(d_theta), 0-np.sin(d_theta), 0], [np.sin(d_theta), np.cos(d_theta), 0], [0, 0, 1]]) # rotation matrix using d_theta, theta > 0 counterclockwise
    
    coord = np.dot(np.matmul(T, R) , np.array([x, y, 1])) # returns a matrix

    coord = coord.getA()[0]
    
    return coord[0:2]

def get_Car_Pos_at_Frame(frame_idx):
    # get lat and lng
    R_GPS = np.matrix([[np.cos(math.pi/2 - yaw[0]), 0-np.sin(math.pi/2 - yaw[0]), 0], [np.sin(math.pi/2 - yaw[0]), np.cos(math.pi/2 - yaw[0]), 0], [0, 0, 1]]) # rotation matrix using d_theta, theta > 0 counterclockwise
    car_pos_diff = np.dot(R_GPS , np.array([position_diff_x[frame_idx], position_diff_y[frame_idx], 1]))
    car_pos_diff = car_pos_diff.getA()[0]
    return car_pos_diff[0:2]

def get_Car_Direction_at_Frame(frame_idx):
    #get yaw from oxts and interpolate
    return yaw[frame_idx]

# For each frame and for each object: Label, score, coordinates, histogram
def calculate_trajectories():
    global data_frames
    pickle.dump( data_frames, open( "data_framesBeforeElab.p", "wb" ) )

    n_frames = len(data_frames)
    id_counter = 0 # use incrementing ints as Object IDs, more human readable

    backward_frames = 4 # number of frames to look in the past to see if an object is already known

    dist_thr = 10
    hist_thr = 0.3

    for i in range(n_frames):
        objs = data_frames[i]
        for o in objs:
            predicted_class = o['predicted_class']
            coords = o['coord']
            hist = o['hist']
            found = False
            for j in range(backward_frames):
                if found:
                    break
                if i-j-1 > 0:
                    old_objs = data_frames[i-j-1]
                    for old_o in old_objs:
                        if old_o['predicted_class'] == predicted_class:
                            coord_dist = np.linalg.norm(coords - old_o['coord'])
                            hist_diff = np.sum(np.linalg.norm(hist - old_o['hist']))/3 #per normalizzare. Sono 3 canali, 3 istogrammi
                            print("Differenza istogrammi")
                            print(hist_diff)
                            if coord_dist < dist_thr and hist_diff < hist_thr:
                                o['id'] = old_o['id']
                                found = True
                                break
            if not found:
                o['id'] = id_counter
                id_counter += 1
    # Now each object has an ID
    pickle.dump( data_frames, open( "data_frames.p", "wb" ) )

def histogram(image, box):
    np_arr = np.asarray(image)
    l_y, l_x, r_y, r_x = box
    box_img = np_arr[int(round(l_y)):int(round(r_y)), int(round(l_x)):int(round(r_x)), :]
    hist_R, b = np.histogram(box_img[:,:,0], 256, (0,255))
    hist_G, b = np.histogram(box_img[:,:,1], 256, (0,255))
    hist_B, b = np.histogram(box_img[:,:,2], 256, (0,255))

    #normalize histograms
    if np.sum(hist_R, dtype=np.int32) is not 0:
        hist_R = hist_R/np.sum(hist_R, dtype=np.int32)
    if np.sum(hist_G, dtype=np.int32) is not 0:
        hist_G = hist_G/np.sum(hist_G, dtype=np.int32)
    if np.sum(hist_B, dtype=np.int32) is not 0:
        hist_B = hist_B/np.sum(hist_B, dtype=np.int32)

    return np.array([hist_R, hist_G, hist_B])

def _main(args):
    model_path = os.path.expanduser(args.model_path)
    start_time = timer()
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    logging.basicConfig(filename=output_path + "/tracking.log", level=logging.DEBUG)

    #parse car positions and angles
    print("Parsing timestamps and oxts files...")
    if args.oxts.startswith('..'):
        parse_oxts(input_path + "/" + args.oxts, input_path + "/" + args.time_stamps)
    else:
        parse_oxts(args.oxts, args.time_stamps)
    print("Done. Data acquired.")

    sess = K.get_session()  # dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    frame_idx = 0

    for image_file in os.listdir(input_path):
        try:
            image_type = imghdr.what(os.path.join(input_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(os.path.join(input_path, image_file))
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        logging.info("Img: " + str(image_file))

        boxes_data = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            box = [max(0,v) for v in box] #sometimes it's negative.
            score = out_scores[i]

            # log positions

            obj_coord = np.array([])

            if predicted_class in ["person", "bicycle", "car", "motorbike", "bus", "train", "truck"] and score > 0.2: #object and classes to track
                if predicted_class in ["bus", "truck"]: #vehicle
                    predicted_class = "car"
                obj_coord = computeCoordinates(box, frame_idx)
                if obj_coord is not None:
                    hist = histogram(image, box)
                    #create data and store it
                    boxes_data.append({
                        'predicted_class': predicted_class,
                        'score': float(score),
                        'coord': obj_coord,
                        'hist': hist
                        })
                    logging.info(predicted_class + " :" + str(obj_coord) + " | " + str(np.linalg.norm(obj_coord)))

            # end log positions
            if saveImages:
                if obj_coord is not None:
                    label = '{} {:.2f} {} {:.2f}'.format(predicted_class, score, str(obj_coord), np.linalg.norm(obj_coord))
                else:
                    label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        frame_idx += 1
        global data_frames
        data_frames.append(boxes_data)
        if saveImages:
            image.save(os.path.join(output_path, image_file), quality=80)

    sess.close()
    now = timer()
    start_trj_time = timer()
    print("Time elapsed CNN: " + str(now - start_time) + " seconds")
    print("Calculating trajectories...")
    calculate_trajectories()

    now = timer()
    print("Done. Time elapsed: " + str(now - start_trj_time) + " seconds\n\n")
    print("Total time elapsed: " + str(now - start_time) + " seconds")

if __name__ == '__main__':
    _main(parser.parse_args())
