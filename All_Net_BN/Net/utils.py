import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask,n_classes=6, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_labels():
    return np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128]])

if __name__ == "__main__":
    # from SegDataFolder import semData
    # trainset = semData(train=True)
    
    # for i in range(10):
    #     arr = decode_segmap(trainset[i]['Y'].numpy())
    #     print(arr.shape)
    #     plt.imsave('checkpoints/{}.jpg'.format(i),arr)
    # from PIL import Image
    # img = Image.open('test_output/end_N-33-130-A-d-4-4_100.jpg')
    # img = np.array(img)
    # print((img!=0).sum())
    import cv2
    a = 255*np.ones((64,64)).astype(np.int8)
    print(a)
    cv2.imwrite('test_output/0.png',a,[int(cv2.IMWRITE_JPEG_QUALITY),95])

    
    # t = cv2.imread('test_output/0.png',cv2.IMREAD_UNCHANGED)
    # print(t.sum())

    t = cv2.imread('/data16/weixian/ladar/Data/test/labels_0-1/14_18.png',cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('test_output/000.png',t*255,[int(cv2.IMWRITE_JPEG_QUALITY),95])
