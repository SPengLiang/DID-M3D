import os
import tqdm

import torch
import numpy as np

import shutil
from datetime import datetime



from tools import eval

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
class Tester(object):
    def __init__(self, cfg_tester, cfg_dataset, model, data_loader, logger):
        self.cfg = cfg_tester
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_dir = cfg_dataset['label_dir']
        self.eval_cls = cfg_dataset['eval_cls']

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg_tester['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=50)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['threshold']
                                     )
            results.update(dets)
            progress_bar.update()

        output_dir = os.path.join(
            self.cfg['out_dir'],
            os.path.basename(os.path.splitext(self.cfg['resume_model'])[0])
        )
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        self.save_results(results, output_dir=output_dir)
        progress_bar.close()
        eval.eval_from_scrach(
            self.label_dir,
            os.path.join(output_dir, 'data'),
            self.eval_cls,
            ap_mode=40)


    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()







