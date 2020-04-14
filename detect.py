import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    if isinstance(img_size, int):
        img_size = (img_size, img_size,)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    #model = Darknet(opt.cfg, img_size)
    model = DarknetInferenceWithNMS(opt.cfg, img_size, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic_nms=opt.agnostic_nms, half=half)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.fuse()
    model.to(device).eval()

    img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
    r = model(img.to(device))

    if TORCHSCRIPT_EXPORT:
        model = torch.jit.script(model)
        #model = torch.jit.trace(model, img)
        model.save(weights+'.pth')
        pass

    # Export mode
    if ONNX_EXPORT:
        r = model(img.to(device))
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11, example_outputs=r)

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[173, 70, 6], [164, 29, 217], [24, 84, 84], [58, 141, 36], [29, 39, 15],
     [169, 103, 78], [199, 153, 252], [191, 14, 143], [33, 30, 106],
     [119, 248, 132], [183, 209, 94], [170, 131, 135], [161, 211, 8],
     [0, 127, 99], [251, 93, 145], [175, 206, 141], [249, 2, 236],
     [133, 242, 232], [106, 53, 33], [133, 203, 190], [133, 201, 20],
     [224, 80, 78], [231, 21, 21], [237, 144, 25], [230, 45, 200],
     [24, 61, 69], [246, 240, 9], [109, 156, 164], [124, 242, 227],
     [87, 161, 8], [101, 115, 167], [187, 116, 201], [228, 238, 218],
     [41, 58, 124], [140, 253, 55], [74, 167, 254], [219, 96, 227],
     [62, 111, 88], [194, 136, 33], [105, 236, 59], [203, 209, 85],
     [189, 44, 27], [141, 242, 45], [234, 99, 24], [67, 184, 109],
     [154, 132, 144], [174, 12, 123], [232, 134, 126], [242, 189, 3],
     [73, 58, 72], [28, 86, 197], [206, 229, 138], [36, 97, 189],
     [92, 24, 128], [237, 98, 239], [176, 30, 200], [215, 141, 36],
     [192, 218, 1], [65, 232, 164], [154, 174, 51], [105, 41, 170],
     [240, 16, 25], [58, 238, 14], [157, 190, 137], [140, 7, 143],
     [102, 66, 74], [141, 129, 170], [179, 82, 50], [168, 58, 180],
     [233, 40, 0], [194, 230, 194], [201, 148, 2], [0, 104, 64],
     [152, 155, 170], [112, 248, 134], [39, 76, 179], [14, 149, 151],
     [241, 104, 21], [76, 21, 180], [189, 16, 41]]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, (rects, cls_ids, cls_scores, all_scores) in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[1:]  # print string
            if len(rects):
                # Rescale boxes from img_size to im0 size
                rect = scale_coords(img.shape[1:], rects, im0.shape).round()

                # Print results
                for c in cls_ids.unique():
                    n = (cls_ids == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for ri, rect in enumerate(rects):
                    cls, conf = cls_ids[ri], cls_scores[ri]
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*rect, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(rect, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
