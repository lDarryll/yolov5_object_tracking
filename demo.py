import sys
sys.path.append('/data/zengsheng/works/work1/eval_algorithms/yolov5_object_tracking/tracker_byte')

from pytz import country_timezones
from AIDetector_pytorch import Detector
import os
import numpy as np
import imutils
from loguru import logger
from sort import Sort
import torch
import cv2

from tracker_byte.byte_tracker import BYTETracker
# from tracker.byte_tracker import BYTETracker
import time
from utils.visualize import plot_tracking
from tracking_utils.timer import Timer
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Object Tracking !")
    parser.add_argument("--choose_tracking", default= 0, type=int, help="choose tracking algorithm, 0:byte_track, 1:deepsort, 2:sort")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default="yolov5s", help="model name")

    parser.add_argument(
        "--path", default="/data/zengsheng/works/work1/eval_algorithms/ByteTrack_yolov5/ADAS_20190525-132500_128_adas.mp4", help="path to images or video"
        # "--path", default="./videos/16h-17h.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default="True",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument("-c", "--ckpt", default="./weights/yolov5s.pt", type=str, help="ckpt for eval")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--num_classes", type=int, default=80, help="number of classes")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(640, 640), type=tuple, help="test image size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    return parser



def byte_track_imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_name = args.save_name
    save_folder = os.path.join(
        vis_folder, save_name
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, save_name + ".mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
        if ret_val:
            # outputs, img_info = predictor.inference(frame, timer)
            _, bboxes = predictor.detect(frame)
            bbox_xyxy = []
            confs = []
            clss = []
            # names = ['bus', 'truck', 'car']
            for x1, y1, x2, y2, cls_id, conf in bboxes:
                # if  cls_id not in names:
                #     print(conf)
                #     print(cls_id)
                #     dets = None
                #     continue
                obj1 = [int(x1), int(y1), int(x2), int(y2), float(conf)]
                bbox_xyxy.append(obj1)
                dets = np.array(bbox_xyxy);
                confs.append(conf)
                clss.append(cls_id)
            # for i, det in enumerate(outputs):
            if len(bbox_xyxy)>0:
                online_targets = tracker.update(dets)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                timer.toc()
                online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = frame
            if args.save_result:
                vid_writer.write(online_im)
            cv2.imshow("online_im", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()

def deepsort_imageflow_demo(predictor, vis_folder, current_time, args):
    name = 'yolov5_deepsort'
    cap = cv2.VideoCapture(args.path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None
    # deep = 3
    while True:
        # try:
        _, im = cap.read()
        # frame_id = 0
        if im is None:
            break
        
        result = predictor.feedCap(im)
        result = result['frame']
            # result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

def sort_imageflow_demo(predictor, vis_folder, current_time, args):
    name = "sort_yolov5"
    tracker = Sort()
    cap = cv2.VideoCapture(args.path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    videoWriter = None
    while True:
        # try:
        _, im = cap.read()
        frame_id = 0
        if im is None:
            break
        
        _, bboxes = predictor.detect(im);
        bbox_xywh = []
        bbox_xyxy = []
        confs = []
        clss = []
        names = ['bus', 'truck', 'car']
        for x1, y1, x2, y2, cls_id, conf in bboxes:
            if conf < 0.5 or cls_id not in names:
                print(conf)
                print(cls_id)
                continue
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            obj1 = [int(x1), int(y1), int(x2), int(y2)]
            bbox_xywh.append(obj)
            bbox_xyxy.append(obj1)
            dets = np.array(bbox_xyxy);
            confs.append(conf)
            clss.append(cls_id)

        if len(bbox_xyxy) == 0:
            result = im
            # result = imutils.resize(result, height=500)
            continue
        else:
            tracks = tracker.update(dets)
            # print("len(tracks):",end=" ")
            # print(len(tracks))
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            for track in tracks:
                bbox = track[:4] # 跟踪框坐标
                indexID = int(track[4]) # 跟踪编号
                # print(indexID)
                # 随机分配颜色
                color = [int(c) for c in COLORS[indexID % len(COLORS)]]
                # 各参数依次是：照片/（左上角，右下角）/颜色/线宽
                cv2.rectangle(im, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
                # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                cv2.putText(im, str(indexID), (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-1, color, 2)
                result = im
                # result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(1)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    


def main(args):

    file_name = os.path.join('runs', '')
    os.makedirs(file_name, exist_ok=True)
    vis_folder = os.path.join(file_name, "track")
    os.makedirs(vis_folder, exist_ok=True)
    current_time = time.localtime()
    det = Detector()
    if(args.choose_tracking == 0):
        print("start bytetrack")
        byte_track_imageflow_demo(det, vis_folder, current_time, args)
    if(args.choose_tracking == 1):
        print("start deepsort")
        deepsort_imageflow_demo(det, vis_folder,current_time, args)
        print("start sort")
    if(args.choose_tracking == 2):
        sort_imageflow_demo(det, vis_folder, current_time, args)
   

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)