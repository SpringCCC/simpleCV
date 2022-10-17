from pprint import pprint

class Config:

    bs = 4
    env = "1017"
    cuda = 1
    dataset_dir = r"E:\ImageData\VOCdevkit\VOC2007\voc2012_forYolov1"
    out_c = 30
    grid_x = 7
    grid_y = 7
    n_cls = 20
    lr = 1e-3
    wd = 0
    epochs = 100

    def _update_params(self, kwargs: dict):
        print("Update params")
        state_dic = self._state_dic()
        for k, v in kwargs.items():
            if k not in state_dic:
                raise ValueError("{} is not in Config, please check keys".format(k))
            else:
                setattr(self, k, v)
        state_dic = self._state_dic()
        pprint(state_dic)

    def _state_dic(self):
        return {k:getattr(self, k) for k, _ in Config.__dict__.items() if k[0]!="_"}



opt =  Config()