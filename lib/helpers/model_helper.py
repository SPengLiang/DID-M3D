from lib.models.DID import DID


def build_model(cfg,mean_size):
    if cfg['type'] == 'DID':
        return DID(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
