import pickle

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DumpResultHook(Hook):

    def __init__(self, interval=1):
        self.interval = interval

    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):

        for i in range(outputs.size(0)):
            data_sample = data_batch['data_samples'][i]
            output = dict(
                occ_pred=outputs[i].cpu().numpy(),
                occ_gt=(data_sample.gt_pts_seg.semantic_seg.squeeze().cpu().
                        numpy()),
                mask_camera=data_sample.mask_camera,
                img_path=data_sample.img_path)

            with open(f'outputs/{data_sample.sample_idx}.pkl', 'wb') as f:
                pickle.dump(output, f)
