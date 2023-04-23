import torch
from torchvision import transforms

import sys
import argparse
import time
import math

sys.path.append('C:/Users/Taras Zykov/OneDrive - Danmarks Tekniske Universitet/DTU/Thesis/code/yolov7')
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pylab as plt
import matplotlib
matplotlib.use('agg')
import cv2
import numpy as np

parser = argparse.ArgumentParser(prog='TestYOLOPose',
                                 description='Test the Yolo Pose model.')
parser.add_argument('-i', '--image', type=bool, default=False)
parser.add_argument('-v', '--video', type=bool, default=False)
parser.add_argument('-c' ,'--camera', type=bool, default=False)
args = parser.parse_args()


# Check that the device has available gpu
if torch.cuda.is_available():
    print(f"GPU available with name {torch.cuda.device.__name__}")
else:
    print("GPU not available, using CPU instead.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    '''
    Function to load the yolov7 pose estimation model.
    '''

    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']

    # Put the model in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predicitons into float16 tensors, which lowers inference time
        model.half().to(device)

    return model

def run_inference(model, image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    
    return output, image

def draw_keypoints(model, output, image):
    output = non_max_suppression_kpt(output,
                                     0.25, #Confidence threshold
                                     0.65, #IoU Threshold
                                     nc=model.yaml['nc'], # Number of classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)
    # Drop
    if len(output) != 0:
        output = output[:, 7:]
    # Fit the frame to needed representation
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    kpts = []
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx].T, 3) # steps will always has to be 3, since for each keypoint we have 3 elements {x, y, conf.}
        kpts.append(output[idx])
    return nimg, kpts

def pose_estimation_image(model, image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Run inference on an image
    output, image = run_inference(model, image)

    
    # Image with skeleton
    nimg, kpts = draw_keypoints(model, output, image)

    # Distance between the 2 shoulders in the image
    for _, kpt in enumerate(kpts):
        print(f"Distance between shoulders: {dist_between_shoulders(kpt)}")
        looking_front = person_front_detection(kpt)
        draw_label(kpt, looking_front, nimg)

    # Plot the keypoints
    plt.figure(figsize=(12,12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.savefig("playground/pose_estimation.png")
    plt.show()

def pose_estimation_video(model, filename):
    cap = cv2.VideoCapture(filename)
    new_frame_time = 0
    prev_frame_time = 0
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret:
            # Take the frame and run it through the model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(model, frame)
            frame, kpts = draw_keypoints(model, output, frame)
            # Calculate the fps
            new_frame_time = time.time()
            fps = 1/(new_frame_time - prev_frame_time)

            # Display if people are looking at the camera or not
            for _, kpt in enumerate(kpts):
                looking_front = person_front_detection(kpt)
                draw_label(kpt, looking_front, frame)

            # After doing stuff: print the fs count
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100,255,0), 3, cv2.LINE_AA)
            # Show video
            cv2.imshow('Pose Estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def pose_estimation_camera(model):
    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(3, 360)

    new_frame_time = 0
    prev_frame_time = 0
    frame_count = 0 
    while True:
        frame_count += 1
        (ret, frame) = cap.read()
        if ret:
            # Take the frame and run it through the model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(model, frame)
            frame, kpts = draw_keypoints(model, output, frame)
            # Calculate the fps
            new_frame_time = time.time()
            fps = 1/(new_frame_time - prev_frame_time)

            # NOTE: The distance actually will be given by the LiDAR
            for _, kpt in enumerate(kpts):
                looking_front = person_front_detection(kpt)
                draw_label(kpt, looking_front, frame)

            # FPS count    
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,255,0), 3, cv2.LINE_AA)

            # Show video
            cv2.imshow('Pose Estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def dist_between_shoulders(kpts):
    print(kpts)
    # Shoulders position
    pos1 = (int(kpts[5*3]), int(kpts[5*3+1])) # Left shoulder 
    pos2 = (int(kpts[6*3]), int(kpts[6*3+1])) # Right shoulder

    return math.floor(np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2))

def person_front_detection(kpts):
    # If ALL the keypoints of the face are visible, the person is probably standing in front of the camera (maybe consider 3/4 visible)
    # If no (or some) face keypoints are visible, then the person must be giving the back to the robot.¨

    vote_front = 0
    if kpts[0*3 + 2] > 0.5:
        vote_front += 1
    if kpts[1*3 + 2] > 0.5: # Confidence of the left eye more than 0.5
        vote_front += 1
    if kpts[2*3 + 2] > 0.5:
        vote_front += 1
    if kpts[3*3 + 2] > 0.5:
        vote_front += 1
    if kpts[4*3 + 2] > 0.5:
        vote_front += 1
    
    # check the votes
    if vote_front >= 4:
        return True
        #print(f"Person {kidx+1} is looking at the robot.")
    else:
        return False
        #print(f"Person {kidx+1} is not looking at the robot.")

def draw_label(kpts, looking_front, frame):
    if looking_front:
        # Draw a rectangle that says whether the person is looking at the camera or not
        cv2.rectangle(frame, (int(kpts[0*3]-60), int(kpts[0*3+1]-90)), (int(kpts[0*3]+90), int(kpts[0*3+1]-130)), (50,205,50), -1)
        cv2.putText(frame, "LOOKING", (int(kpts[0*3]-50), int(kpts[0*3+1]-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    else:
        # Draw a rectangle that says whether the person is looking at the camera or not
        cv2.rectangle(frame, (int(kpts[0*3]-60), int(kpts[0*3+1]-90)), (int(kpts[0*3]+170), int(kpts[0*3+1]-130)), (50,205,50), -1)
        cv2.putText(frame, "NOT LOOKING", (int(kpts[0*3]-50), int(kpts[0*3+1]-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


def main():
    '''
    Main of the code
    '''
    # Load the model
    model = load_model()

    if args.image:
        pose_estimation_image(model, "playground/robot_image.png")
    elif args.video:
        pose_estimation_video(model, "playground/robot_video.mp4")
    elif args.camera:
        pose_estimation_camera(model)



if __name__ == "__main__":
    main()
