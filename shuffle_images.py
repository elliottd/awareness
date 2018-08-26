import numpy as np
np.random.seed(22052018)  # for reproduction by other researchers
import sys
import argparse


def create_random_shuffle(image_order, image_feats):
    '''
    Randomly shuffles the elements in the image_order list,
    and then realises this random shuffle in the image features ndarray.

    Returns:
        shuffled_order: the randomly shuffled order of the images
        shuffled_feats: the randomly shuffled numpy array
    '''
    indices = list(range(len(image_order)))
    order = dict((k, v) for k, v in zip(indices, image_order))
    shuffled_order = np.random.permutation(indices)  # shuffle and return

    shuffled_feats = np.zeros((image_feats.shape[0], image_feats.shape[1])).astype('float32')
    shuffled_names = []

    # Create the shuffled image features ndarray and the associated names
    for idx, x in enumerate(shuffled_order):
        shuffled_feats[idx] = image_feats[x]
        shuffled_names.append(order[x])
     
    return shuffled_names, shuffled_feats


def create_zero_vectors(num_samples, num_feats):
    '''
    Creates and returns a zero-initialised num_samples X num_feats matrix.

    Returns:
        the num_samples X num_feats matrix
    '''
    
    return np.zeros((num_samples, num_feats)).astype('float32')


def write_shuffled_filenames(names, i, image_order_filename): 
    with open('{}-random{}'.format(image_order_filename, i), 'w') as handle:
        for n in names:
            handle.write('{}\n'.format(n))


def main(args):
    image_order = [x.strip() for x in open(args.image_order_file).readlines()]
    image_feats = np.load(args.features_file)

    zeros = create_zero_vectors(image_feats.shape[0], image_feats.shape[1])
    np.save(open('{}-zeros'.format(args.features_file), 'wb'),
            zeros,
            allow_pickle=False)

    for i in range(args.num_shuffles):
        names, feats = create_random_shuffle(image_order, image_feats)
        np.save(open('{}-random{}'.format(args.features_file, i), 'wb'),
                feats,
                allow_pickle=False)
        write_shuffled_filenames(names, i, args.image_order_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_order_file",
                        help="Path to file containing the order in which the\
                              images are supposed to be processed",
                        type=str,
                        required=True)
    parser.add_argument("--features_file",
                        help="Path to file containing the image features\
                              corresponding to the image_order_file",
                        type=str,
                        required=True)
    parser.add_argument("--num_shuffles", type=int, default=5,
                        help="Number of random shuffles")
    main(parser.parse_args())
