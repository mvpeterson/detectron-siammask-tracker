import os

import pandas as pd
import yaml
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from experiments.siammask_sharp.custom import Custom
from tools.test import *

from config_siammask_helper import *
from object_tracker import *

# print(os.environ["PYTHONPATH"])

VIDEO_INPUT = '/mnt/fileserver/shared/datasets/streetscene/test.mp4'
VIDEO_INPUT = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/meeting/meeting_set005_00:36:30-00:36:30.mp4'
DETECTRON_CONFIG = './configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml'
DETECTRON_WEIGHTS = '/mnt/fileserver/shared/models/detectron/model_0039999_e76410.pkl'
SIAMMASK_WEIGHTS = '/mnt/fileserver/shared/models/SiamMask/siammask_sharp/SiamMask_DAVIS.pth'
SIAMMASK_CONFIG = './configs/siammask_sharp/config_davis.json'
VIDEO_OUTPUT = './video_output.mp4'

detectron_only = True
maxDissapear = 12 #Maximum number of frames for object to be considered as disssapeared

def setup_detectron_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.detectron_config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = args.detectron_weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.detectron_confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.detectron_confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.detectron_confidence_threshold
    cfg.freeze()
    return cfg

def load_config(file):
    ''' This function loads .yml file to Namespace so we can access attributes by name.'''
    with open(file, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    config = argparse.Namespace(**y)
    if not hasattr(config, 'grad_penalty'):
        config.grad_penalty = 0.0
    return config

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--detectron-config-file",
        default=DETECTRON_CONFIG,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", default = VIDEO_INPUT,
                        help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        # "--output",
        "--output", default=VIDEO_OUTPUT,
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--detectron-confidence-threshold",
        type=float,
        default=0.70,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument("--detectron-weights", default=DETECTRON_WEIGHTS)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--siammask-resume', default=SIAMMASK_WEIGHTS, type=str, help='path to latest siammask checkpoint')

    parser.add_argument('--siammask-config', default=SIAMMASK_CONFIG,
                        help='hyper-parameter of SiamMask in json format')


    return parser


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def visualize_result(frame, predictions):
    cpu_device = torch.device("cpu")
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            frame, panoptic_seg.to(cpu_device), segments_info
        )
    elif "instances" in predictions:
        predictions = predictions["instances"].to(cpu_device)
        vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
    elif "sem_seg" in predictions:
        vis_frame = video_visualizer.draw_sem_seg(
            frame, predictions["sem_seg"].argmax(dim=0).to(cpu_device)
        )

    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame

def process_frame(frame, detector):
    # frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
    vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    predictions = detector(vis_frame)
    predictions_instances = predictions["instances"].to(torch.device("cpu"))
    vis_frame = visualize_result(vis_frame, predictions)

    return vis_frame, predictions_instances


def location2bbox(location):
    minX = np.amin(location[:,:,0])
    maxX = np.amax(location[:,:,0])
    minY = np.amin(location[:, :, 1])
    maxY = np.amax(location[:, :, 1])

    bbox = np.array([minX, minY, maxX, maxY])
    return bbox



if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg_detectron = setup_detectron_cfg(args)
    cfg_siammask = load_siammask_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    video = cv2.VideoCapture(args.video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(args.video_input)

    print(video)

    print("Video status:\nwidth: %i\nheight: %i\nfps: %i\nnumber of frames: %i\n" %
          (width, height, frames_per_second, num_frames))

    if args.output:
        if os.path.isdir(args.output):
            output_fname = os.path.join(args.output, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mkv"
        else:
            output_fname = args.output
            output_dfname = args.output + ".csv"
        assert not os.path.isfile(output_fname), "file: " + output_fname + " is already exists"
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )

    assert os.path.isfile(args.video_input)

    detector = DefaultPredictor(cfg_detectron)

    siammask = Custom(anchors=cfg_siammask['anchors'])
    if args.siammask_resume:
        assert isfile(args.siammask_resume), 'Please download {} first.'.format(args.siammask_resume)
        siammask = load_pretrain(siammask, args.siammask_resume)

    siammask.eval().to(device)

    frame_gen = _frame_from_video(video)

    metadata = MetadataCatalog.get(
        cfg_detectron.DATASETS.TEST[0] if len(cfg_detectron.DATASETS.TEST) else "__unused"
    )

    video_visualizer = VideoVisualizer(metadata, instance_mode=ColorMode.IMAGE)


    if detectron_only:
        maxDissapear = 1
    objectTracker = ObjectTracker(maxDissapear)

    frame_idx = 0

    df = pd.DataFrame(columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])

    for frame in frame_gen:

        vis_frame, predictions = process_frame(frame, detector)
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        input_objects = []
        n = boxes.shape[0]
        for i in range(n):
            obj = Object()
            obj.bbox = boxes[i,:]
            obj.score = scores.data[i].item()
            obj.label = classes[i]
            input_objects.append(obj)

        tracker_objects = objectTracker.update(input_objects)
        objectIDs = list(tracker_objects.keys())
        n = len(tracker_objects)


        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        fontColor = (255, 0, 0)
        lineType = 2

        for id in objectIDs:

            x1 = tracker_objects[id].bbox[0]
            y1 = tracker_objects[id].bbox[1]
            x2 = tracker_objects[id].bbox[2]
            y2 = tracker_objects[id].bbox[3]
            w = (x2 - x1)
            h = (y2 - y1)
            target_pos = np.array([x1 + w / 2, y1 + h / 2])
            target_sz = np.array([w, h])

            if not detectron_only:
                if tracker_objects[id].isdissapeared:
                    if tracker_objects[id].trackingStatus == False:
                        tracker_objects[id].state = siamese_init(objectTracker.framebuf, target_pos, target_sz, siammask,
                                                                cfg_siammask['hp'],
                                                                device=device)
                        tracker_objects[id].trackingStatus = True


                    tracker_objects[id].state = siamese_track(tracker_objects[id].state, frame, mask_enable=True,
                                                             refine_enable=True, device=device)
                    location = tracker_objects[id].state['ploygon'].flatten()
                    tracker_objects[id].location = np.int0(location).reshape((-1, 1, 2))
                    tracker_objects[id].bbox = location2bbox(tracker_objects[id].location)

            x1 = tracker_objects[id].bbox[0]
            y1 = tracker_objects[id].bbox[1]
            x2 = tracker_objects[id].bbox[2]
            y2 = tracker_objects[id].bbox[3]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_frame, '%d'%id, (x1, y1), font, fontScale, fontColor, lineType)

            # ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']
            df.loc[len(df)] = [frame_idx, id, x1, y1, (x2-x1), (y2-y1), tracker_objects[id].score,
                               tracker_objects[id].label, not tracker_objects[id].trackingStatus]


        cv2.imshow('',vis_frame)
        cv2.waitKey(10)

        if args.output:
            output_file.write(vis_frame)

        frame_idx += 1
        if frame_idx > 30:
            break

        objectTracker.framebuf = frame

        print('%d/%d'%(frame_idx,num_frames))

    if args.output:
        output_file.release
        df.to_csv(output_dfname, index=False)