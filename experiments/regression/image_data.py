import datasets
import tensorflow as tf
from neural_diffusion_processes.types import Batch
from functools import partial
import math


def unflatten_image(flattened_image: tf.Tensor, orig_image_shape: tf.TensorShape) -> tf.Tensor:
    """
    Utility function to unflatten an image (such as a predicted image); to do this the original
    image size must be known

    :param flattened_image: images that we wish to unflatten, assumes it has a leading dimension
    :param orig_image_shape: shape with
    """
    num_pixels_x, num_pixels_y, num_channels = orig_image_shape
    unflattened_image = tf.transpose(flattened_image, (0, 2, 1))
    unflattened_image = tf.reshape(unflattened_image, (-1, num_channels, num_pixels_y, num_pixels_x))
    unflattened_image = tf.transpose(unflattened_image, (0, 3, 2, 1))
    return unflattened_image


def _flatten_images(image: tf.Tensor):
    """ utility function to flatten images, assuming batch dimension """
    num_batches, num_pixels_x, num_pixels_y, num_channels = get_image_info(image)
    flat_image = tf.transpose(image, (0, 3, 2, 1))
    flat_image = tf.reshape(flat_image, (num_batches, num_channels, num_pixels_x*num_pixels_y))
    flat_image = tf.transpose(flat_image, (0, 2, 1))
    return flat_image


def get_image_info(image: tf.Tensor):
    """ Unpack shape of image into constituent parts """
    num_channels = tf.shape(image)[-1]
    num_pixels_y = tf.shape(image)[-2]
    num_pixels_x = tf.shape(image)[-3]
    num_batches = tf.shape(image)[-4]
    return num_batches, num_pixels_x, num_pixels_y, num_channels


def normalise_image(example, pixel_mean: float = 0.0, pixel_std: float = 1.0) -> tf.Tensor:
    """ Convert image values to be [-1, 1] """
    image = example['image']
    converted_image = tf.cast(image, tf.float32)
    normalised_image = converted_image / 255
    normalised_image = 2.0 * (normalised_image - 0.5)
    normalised_image = (normalised_image - pixel_mean) / pixel_std
    example['image'] = normalised_image
    return example


def add_image_channel_if_missing(example):
    """ Add a channel dimension if it doesn't already exist """
    image = example['image']
    if tf.rank(image) == 3:
        example['image'] = image[..., None]
    return example


def flatten_images(example):
    """ Flatten image """
    image = example['image']
    flattened_image = _flatten_images(image)
    example['flat_image'] = flattened_image
    return example


def create_xy_inputs(example):
    """
    Create X, Y locations of each pixel of an image. Repeat for each batch and add to

    Note these image pixel locations are assumed to be within the range [0, 1]
    """
    image = example['image']
    num_batches, num_pixels_x, num_pixels_y, num_channels = get_image_info(image)
    xx_yy_grid_flat = create_xy_grid_features_from_single_image(num_pixels_x, num_pixels_y)
    # Repeat for every batch
    xx_yy_grid_flat = tf.repeat(xx_yy_grid_flat[None, ...], num_batches, axis=0)
    example['xx_yy'] = xx_yy_grid_flat
    return example


def create_xy_grid_features_from_single_image(num_pixels_x: int, num_pixels_y: int) -> tf.Tensor:
    """ Create input features by creating a meshgrid and flattening """
    centred_xx_grid, centred_yy_grid = create_xy_meshgrid(num_pixels_x, num_pixels_y)
    xx_yy_grid_flat = tf.stack([
        tf.reshape(centred_yy_grid, (-1,)),
        tf.reshape(centred_xx_grid, (-1,)),
    ], axis=1)  # [num_pixels_x*num_pixels_y, 2]
    xx_yy_grid_flat = tf.cast(xx_yy_grid_flat, tf.float32)
    return xx_yy_grid_flat


def create_xy_meshgrid(num_pixels_x, num_pixels_y, minval=-5, maxval=5):
    """ Create an x-y meshgrid, where x=0, y=0 is the bottom left hand pixel of the image"""
    centred_x_grid = tf.linspace(start=maxval, stop=minval, num=num_pixels_x)
    centred_y_grid = tf.linspace(start=minval, stop=maxval, num=num_pixels_y)
    # assert centred_x_grid.shape[0] == num_pixels_x
    # assert centred_y_grid.shape[0] == num_pixels_y
    centred_xx_grid, centred_yy_grid = tf.meshgrid(centred_x_grid, centred_y_grid)
    return centred_xx_grid, centred_yy_grid


def split_into_target_and_context(example, number_of_context=(0.9, 0.8, 0.5)):
    total_num_pixels = tf.shape(example['flat_image'])[-2]
    choice_indices = tf.range(len(number_of_context))
    random_choice_index = tf.random.shuffle(choice_indices)[0]
    percent_context_pixels = tf.gather(number_of_context, random_choice_index)
    shuffled_pixel_indices = tf.random.shuffle(tf.range(total_num_pixels))

    context_num_pixels = tf.cast(tf.floor(
        tf.cast(total_num_pixels, tf.float32) * percent_context_pixels
    ), tf.int32)
    context_pixel_indices = shuffled_pixel_indices[:context_num_pixels]
    target_pixel_indices = shuffled_pixel_indices[context_num_pixels:]

    example['x_context'] = tf.gather(example['xx_yy'], context_pixel_indices, axis=-2)
    example['x_target'] = tf.gather(example['xx_yy'], target_pixel_indices, axis=-2)

    example['y_context'] = tf.gather(example['flat_image'], context_pixel_indices, axis=-2)
    example['y_target'] = tf.gather(example['flat_image'], target_pixel_indices, axis=-2)
    return example


def delete_unused_columns(example):
    del example['image']
    del example['flat_image']
    del example['xx_yy']
    return example


def add_mask(example):
    num_batches = tf.shape(example['x_target'])[0]
    num_target = tf.shape(example['x_target'])[1]
    num_context = tf.shape(example['x_context'])[1]
    example['mask_target'] = tf.zeros((num_batches, num_target))
    example['mask_context'] = tf.zeros((num_batches, num_context))
    return example


def get_image_data(
    dataset_name: str = "lansinuote/gen.1.celeba",
    image_col: str = "image",
    batch_size: int = 1024,
    num_epochs: int = 1,
    train: bool = True,
):
    if train:
        subset = 'train'
        #split_batch_into_target_and_context = partial(split_into_target_and_context, number_of_context=(0.0,))
        split_batch_into_target_and_context = partial(split_into_target_and_context, number_of_context=(0.9, 0.7, 0.5))
    else:
        subset = 'test'
        split_batch_into_target_and_context = split_into_target_and_context

    if 'mnist' in dataset_name:
        # NOTE - this is normalised for the average pixel values for MNIST across the whole dataset
        print("normalising")
        pixel_mean = -0.7409841
        pixel_std = math.sqrt(0.38553977)
        normalise_image_map = partial(normalise_image, pixel_mean=pixel_mean, pixel_std=pixel_std)
    else:
        print("WARNING: No mean and std calculated")
        normalise_image_map = normalise_image

    images_dataset = datasets.load_dataset(dataset_name)
    images_dataset.set_format('tensorflow')
    images_dataset = images_dataset.select_columns(image_col)
    images_tf_dataset = images_dataset[subset].to_tf_dataset(batch_size=batch_size, shuffle=True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # NOTE: This this will revisit the data in the same order in each epoch, but will still have
    # random masking for each which will differ between epochs
    images_tf_dataset = images_tf_dataset.repeat(count=num_epochs)
    processed_images_tf_dataset = images_tf_dataset.map(add_image_channel_if_missing)
    processed_images_tf_dataset = processed_images_tf_dataset.map(normalise_image_map)
    processed_images_tf_dataset = processed_images_tf_dataset.map(flatten_images)
    processed_images_tf_dataset = processed_images_tf_dataset.map(create_xy_inputs)
    processed_images_tf_dataset = processed_images_tf_dataset.map(split_batch_into_target_and_context)
    processed_images_tf_dataset = processed_images_tf_dataset.map(delete_unused_columns)
    #processed_images_tf_dataset = processed_images_tf_dataset.map(add_mask)
    processed_images_tf_dataset = processed_images_tf_dataset.prefetch(AUTOTUNE)
    processed_images_tf_dataset = processed_images_tf_dataset.as_numpy_iterator()
    processed_celeb_batch_dataset = map(lambda d: Batch(**d), processed_images_tf_dataset)
    return processed_celeb_batch_dataset


if __name__ == '__main__':
    dataset_name = "mnist"
    print(dataset_name)
    image_data = get_image_data(dataset_name=dataset_name, train=True, batch_size=1000)
    for i in range(10):
        data = next(image_data)
        print(f"x_target: {data.x_target.shape}")
        print(f"y_target: {data.y_target.shape}")
        print(f"x_context: {data.x_context.shape}")
        print(f"y_context: {data.y_context.shape}")
        if data.mask_target is not None:
            print(f"mask_target: {data.mask_target.shape}")
            print(f"mask_context: {data.mask_context.shape}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        # ax.scatter(
        #     data.x_context[0, :, 0],
        #     data.x_context[0, :, 1],
        #     c=data.y_context[0, :, 0],
        # )
        ax.scatter(
            data.x_target[0, :, 0],
            data.x_target[0, :, 1],
            c=data.y_target[0, :, 0],
        )
        print(data.y_target.mean())
        print(data.y_target.var())
        fig.show()
