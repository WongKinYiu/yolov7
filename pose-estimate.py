import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import Colors, plot_one_box_kpt


@torch.no_grad()
def run(opt):
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps
    time_list = []  # list to store time
    fps_list = []  # list to store fps

    device = select_device(opt.device)  # select device
    half = device.type != 'cpu'
    model = attempt_load(opt.poseweights, map_location=device)  # Load model
    if half:
        model.half()  # to FP16
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if opt.source.isnumeric():
        cap = cv2.VideoCapture(int(opt.source))  # pass video to videocapture object
    else:
        cap = cv2.VideoCapture(opt.source)  # pass video to videocapture object

    if not cap.isOpened():  # check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  # get video frame width
        vid_write_image = letterbox(cap.read()[1], frame_width, stride=64, auto=True)[0]  # init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{opt.source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"results/{out_video_name}_yolov7.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (resize_width, resize_height))
        colors = Colors()

        while cap.isOpened:  # loop until cap opened or video not complete
            print("Frame {} Processing".format(frame_count + 1))
            ret, frame = cap.read()  # get frame and success from video capture
            if ret:  # if success is true, means frame exist
                orig_image = frame  # store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
                image = letterbox(image, frame_width, stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)  # convert image data to device
                image = image.float()  # convert image to float precision (cpu)
                start_time = time.time()  # start time for fps calculation

                with torch.no_grad():  # get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                      0.25,  # Conf. Threshold.
                                                      0.65,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                      kpt_label=True)

                im0 = image[0].permute(1, 2,
                                       0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)

                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)

                for i, pose in enumerate(output_data):  # detections per image

                    if len(output_data):  # check if no pose
                        for c in pose[:, 5].unique():  # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))

                        for det_index, (*xyxy, conf, cls) in enumerate(
                                reversed(pose[:, :6])):  # loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (
                                names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=opt.line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                             orig_shape=im0.shape[:2], kpt_thickness=opt.kpt_thickness)

                end_time = time.time()  # Calculation for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                fps_list.append(total_fps)  # append FPS in list
                print(f'FPS avg: {total_fps / (frame_count + 1)}')
                time_list.append(end_time - start_time)  # append time in list

                # Stream results
                if opt.view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  # writing the video frame

                print(f'FPS: {(frame_count + 1) / sum(time_list)}')
            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

        # plot the comparison graph
        plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='case2_axis4_00-04-19_00-05-24.mp4', help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arguments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true', default='store_true',
                        help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')  # box line thickness
    parser.add_argument('--kpt-thickness', default=6, type=int,
                        help='bounding box thickness (pixels)')  # pose line thickness | pose key point circle radius = line-thickness + 1
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hide label
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # box hide conf
    opt = parser.parse_args()
    return opt


# function for plot fps and time comparison graph
def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparson Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparison_pose_estimate.png")


if __name__ == "__main__":
    args = parse_opt()
    print(args)
    run(args)
