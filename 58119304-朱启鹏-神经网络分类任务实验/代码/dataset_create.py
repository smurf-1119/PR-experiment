import mindspore.dataset.vision.c_transforms as vi_transforms
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype


def dataset_create(sample_num, data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    '''
    1.载入数据集
    2.增强数据
    3.对数据集打乱，并返回
    '''
    image_dataset = ds.Cifar10Dataset(data_path, num_samples=sample_num, shuffle=True)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081
    resize_op = vi_transforms.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = vi_transforms.Rescale(rescale_nml, shift_nml)
    rescale_op = vi_transforms.Rescale(rescale, shift)
    hwc2chw_op = vi_transforms.HWC2CHW()
    type_cast_op = c_transforms.TypeCast(mstype.int32)
    image_dataset = image_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    image_dataset = image_dataset.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    image_dataset = image_dataset.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    image_dataset = image_dataset.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    image_dataset = image_dataset.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    buffer_size = 10000
    image_dataset = image_dataset.shuffle(buffer_size=buffer_size)
    image_dataset = image_dataset.batch(batch_size, drop_remainder=True)
    return image_dataset