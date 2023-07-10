from torch.nn import DataParallel

class UNetDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        # This is because the timestep (inputs[1]) in UNet is a 0-d tensor and scatter will try to split inputs[1]. We simply convert it to a float so that scatter has no effect on it.
        
        inputs = inputs[0], inputs[1].item()
        return super().forward(*inputs, **kwargs)
