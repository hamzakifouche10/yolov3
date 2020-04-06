import torch
import time
from models import Darknet
from utils.datasets import LoadImages

def run_test(model, dataset, device):
    last_img = None
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0
        img = img.unsqueeze(0)

        # Inference
        last_img = img
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            pred = model(img)
        torch.cuda.synchronize()
        t2 = time.time()
        print('Done. (%.3fs)' % (t2 - t1))
    return last_img



def run():
    img_size = 416
    cfg, weights = ('cfg/yolov3-tiny.cfg', 'weights/yolov3-tiny.pt') # simpler model
    #cfg, weights = ('cfg/yolov3.cfg', 'weights/yolov3.pt') # more complex model
    device = 'cuda:0'
    source = 'data/samples'

    # Initialize model
    print(cfg, weights)
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    dataset = LoadImages(source, img_size=img_size)

    print('before scripting:')
    img = run_test(model, dataset, device)

    print('after scripting:')
    model = torch.jit.script(model)
    img = run_test(model, dataset, device)

if __name__ == '__main__':
    assert torch.cuda.is_available()
    run()
