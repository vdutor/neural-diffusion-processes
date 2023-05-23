import datasets
import tensorflow as tf
from neural_diffusion_processes.types import Batch


def unflatten_image(flattened_image: tf.Tensor, orig_image_shape: tf.TensorShape) -> tf.Tensor:
    num_pixels_x, num_pixels_y, num_channels = orig_image_shape
    unflattened_image = tf.transpose(flattened_image, (1, 0))
    unflattened_image = tf.reshape(unflattened_image, (num_channels, num_pixels_y, num_pixels_x))
    unflattened_image = tf.transpose(unflattened_image, (2, 1, 0))
    return unflattened_image


def flatten_images(image: tf.Tensor):
    """ flatten images, assuming batch dimension """
    num_batches, num_pixels_x, num_pixels_y, num_channels = get_image_info(image)
    flat_image = tf.transpose(image, (0, 3, 2, 1))
    flat_image = tf.reshape(flat_image, (num_batches, num_channels, num_pixels_x*num_pixels_y))
    flat_image = tf.transpose(flat_image, (0, 2, 1))
    return flat_image


def get_image_info(image: tf.Tensor):
    num_batches = tf.shape(image)[0]
    num_pixels_x = tf.shape(image)[1]
    num_pixels_y = tf.shape(image)[2]
    num_channels = tf.shape(image)[3]
    return num_batches, num_pixels_x, num_pixels_y, num_channels


def normalise_image(image: tf. Tensor) -> tf.Tensor:
    converted_image = tf.cast(image, tf.float32)
    normalised_image = converted_image / 255
    normalised_image = 2.0 * (normalised_image - 0.5)
    return normalised_image


def process_data(example):
    image = example['image']
    normalised_image = normalise_image(image)
    flattened_normalised_image = flatten_images(normalised_image)
    example['flat_image'] = flattened_normalised_image
    return example


def create_xy_data(example):
    image = example['image']
    num_batches, num_pixels_x, num_pixels_y, num_channels = get_image_info(image)

    # make xys
    centred_x_grid = tf.linspace(start=1, stop=-1, num=num_pixels_x)
    centred_y_grid = tf.linspace(start=-1, stop=1, num=num_pixels_y)
    # assert centred_x_grid.shape[0] == num_pixels_x
    # assert centred_y_grid.shape[0] == num_pixels_y
    centred_xx_grid, centred_yy_grid = tf.meshgrid(centred_x_grid, centred_y_grid)
    xx_yy_grid_flat = tf.stack([
        tf.reshape(centred_yy_grid, (-1,)),
        tf.reshape(centred_xx_grid, (-1,)),
    ], axis=1)  # [num_pixels_x*num_pixels_y, 2]
    xx_yy_grid_flat = tf.cast(xx_yy_grid_flat, tf.float32)
    xx_yy_grid_flat = tf.repeat(xx_yy_grid_flat[None, ...], num_batches, axis=0)
    example['xx_yy'] = xx_yy_grid_flat
    return example


def split_into_target_and_context(example, number_of_context=(0.1, 0.2, 0.5)):
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


def get_image_data(
        dataset_name: str = "lansinuote/gen.1.celeba",
        image_col: str = "image",
        batch_size: int = 1024,
        num_epochs: int = 1,
):
    celeb_dataset = datasets.load_dataset(dataset_name)
    celeb_dataset.set_format('tensorflow')
    celeb_images_dataset = celeb_dataset.select_columns(image_col)
    celeb_tf_dataset = celeb_images_dataset['train'].to_tf_dataset(batch_size=batch_size, shuffle=True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # NOTE: This this will revisit the data in the same order in each epoch, but will still have
    # random masking for each which will differ between epochs
    celeb_tf_dataset = celeb_tf_dataset.repeat(count=num_epochs)
    processed_celeb_tf_dataset = celeb_tf_dataset.map(process_data)
    processed_celeb_tf_dataset = processed_celeb_tf_dataset.map(create_xy_data)
    processed_celeb_tf_dataset = processed_celeb_tf_dataset.map(split_into_target_and_context)
    processed_celeb_tf_dataset = processed_celeb_tf_dataset.map(delete_unused_columns)
    processed_celeb_tf_dataset = processed_celeb_tf_dataset.prefetch(AUTOTUNE)
    processed_celeb_tf_dataset = processed_celeb_tf_dataset.as_numpy_iterator()
    return map(lambda d: Batch(**d), processed_celeb_tf_dataset)


if __name__ == '__main__':
    image_data = get_image_data()
    for i in range(10):
        data = next(image_data)
        print(data.keys())
        print(data['x_target'].shape)
        print(data['y_target'].shape)
        print(data['x_context'].shape)
        print(data['y_context'].shape)
