import utils
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_decay_full(TEs: npt.NDArray, signals: npt.NDArray, title: str) -> None:
    if len(signals.shape) == 1:
        signals = np.array([signals])

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    for voxel in range(signals.shape[0]):
        log_signal = np.log(signals[voxel])
        y = log_signal.reshape(len(TEs), 1)
        x = np.array([np.ones((len(TEs), 1)), TEs.reshape(len(TEs), 1)]).T
        b = np.linalg.pinv(x) @ y
        y_pred_linear = np.squeeze(x @ b)

        axes[0].set_title("Unscaled")
        axes[0].plot(TEs, signals[voxel], "o-", color="blue")
        axes[0].plot(TEs, np.exp(y_pred_linear), "x--", color="red")
        axes[0].set_ylim(0, 3800)
        axes[0].set_ylabel("Signal")
        axes[0].set_xlabel("Echo time")
        axes[1].plot(TEs, log_signal, "o-")
        axes[1].plot(TEs, y_pred_linear, "x--")
        axes[1].set_title("Log scaled, centered")
        axes[1].set_ylabel("Signal")
        axes[1].set_xlabel("Echo time")
    fig.suptitle(title)
    fig.savefig(f"images/{title}.png")
    #plt.show()

def plot_decay(TEs: npt.NDArray, signals: npt.NDArray, title: str) -> None:
    if len(signals.shape) == 1:
        signals = np.array([signals])

    fig, axes = plt.subplots(figsize=(5,5))
    for voxel in range(signals.shape[0]):
        log_signal = np.log(signals[voxel])
        y = log_signal.reshape(len(TEs), 1)
        x = np.array([np.ones((len(TEs), 1)), TEs.reshape(len(TEs), 1)]).T
        b = np.linalg.pinv(x) @ y
        y_pred_linear = np.squeeze(x @ b)

        axes.plot(TEs, log_signal, "o-", label="Actual decay", color="blue")
        axes.plot(TEs, y_pred_linear, "x--", label="Predicted exponential decay", color="red")
    axes.set_title(title)
    axes.set_ylabel("Log signal")
    axes.set_xlabel("Echo time")
    #axes.set_ylim(5.5, 8.3)
    axes.legend()
    #fig.tight_layout()
    fig.savefig(f"images/{title}.png")
    #plt.show()


def get_binary_mask(mask: npt.NDArray) -> npt.NDArray:
    zeros = np.zeros(mask.shape)
    ones = np.ones(mask.shape)
    positive_mask = mask
    if (mask.min() < 0):
        positive_mask = mask - mask.min()
    norm_mask = positive_mask / positive_mask.max()
    return np.where(norm_mask > 0.99, ones, zeros)


def plot_masked_decay(slice: npt.NDArray, segmentation_mask: npt.NDArray, title: str) -> None:
    binary_mask = get_binary_mask(segmentation_mask)
    masked = slice[binary_mask.nonzero()]
    masked_mean = masked.mean(axis=0)
    plot_decay(TEs, masked_mean, title)


def plot_volume_masked_decay(signal: npt.NDArray, segmentation_masks: npt.NDArray, title: str) -> None:
    slice_means = np.zeros((signal.shape[2], len(TEs)))
    for i in range(signal.shape[2]):
        slice = signal[:, :, i]
        segmentation_mask = segmentation_masks[:, :, i]
        if not np.any(segmentation_mask):
            continue
        binary_mask = get_binary_mask(segmentation_mask)
        masked = slice[binary_mask.nonzero()]
        masked_mean = masked.mean(axis=0)
        slice_means[i] = masked_mean
    
    plot_decay(TEs, slice_means.mean(axis=0), title)


TEs = utils.get_te_times()

img_id = "11610"
signal = utils.load_signal(img_id=img_id)
signal_reg = utils.load_reg(img_id=img_id)
mask = utils.load_mask(img_id=img_id)
segmentations = utils.load_segmentations(img_id=img_id)

voxel_slice = 25
voxels = np.asarray([[38, 35, voxel_slice],
                    [42, 60, voxel_slice],
                    [47, 69, voxel_slice],
                    [53, 50, voxel_slice]])

voxel_signals = signal[voxels[:,0], voxels[:,1], voxels[:,2]]
plot_decay(TEs, voxel_signals, "Signal decay for single voxes")

column_means_across_slices = signal[:, :].mean(axis=2)
column_means = column_means_across_slices[voxels[:,0], voxels[:,1]]
plot_decay(TEs, column_means, "Signal decay for voxels average across slices")

all_dimension_mean = column_means_across_slices.mean(axis=(0,1))
plot_decay(TEs, all_dimension_mean, "Signal decay averaged across all voxels")

slice_signal = signal[:, :, voxel_slice]
slice_segmentations = segmentations[:, :, voxel_slice]

unmasked_mean = slice_signal.mean(axis=(0,1))
plot_decay(TEs, unmasked_mean, "Mean signal decay for unmasked slice")

simplified = slice_signal[np.nonzero(mask[:, :, voxel_slice])]
masked_mean = simplified.mean(axis=0)
plot_decay(TEs, masked_mean, "Mean signal decay for masked slice")

plot_volume_masked_decay(signal, segmentations[:, :, :, 1], "Cerebro-spinal fluid")
plot_volume_masked_decay(signal, segmentations[:, :, :, 2], "Grey matter")
plot_volume_masked_decay(signal, segmentations[:, :, :, 3], "White matter")
plot_volume_masked_decay(signal, segmentations[:, :, :, 4], "Deep grey matter")
