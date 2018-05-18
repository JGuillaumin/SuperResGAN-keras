import numpy as np
from tensorflow.python.client import device_lib

def print_available_devices():
    local_device_protos = [(x.name, x.device_type, x.physical_device_desc)  for x in device_lib.list_local_devices()]
    for device_name, device_type, device_desc in local_device_protos:
        print("Device : {}\n\t type : {}\n\t desc :{}\n".format(device_name, device_type, device_desc))

def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x,dtype=np.float32)


def deprocess_HR(x):
    return np.clip((x+np.ones_like(x))*127.5, 0, 255)


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    return np.clip(x*255, 0, 255)