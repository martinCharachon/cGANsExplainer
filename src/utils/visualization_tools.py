import matplotlib.pyplot as plt
import numpy as np


def show_multi_visualization(
        viz_list: list = None,
        image_list: list = None,
        annotations_list: list = None,
        title=None):
    if viz_list is not None:
        if image_list is not None:
            if len(viz_list) > len(image_list):
                num_images = len(viz_list)
                while len(image_list) != len(viz_list):
                    image_list.append(None)
            elif len(viz_list) < len(image_list):
                num_images = len(image_list)
                while len(viz_list) != len(image_list):
                    viz_list.append(None)
            elif len(viz_list) == len(image_list):
                num_images = len(image_list)
            else:
                raise KeyError
        else:
            num_images = len(viz_list)
            image_list = [None for i in range(num_images)]

    else:
        if image_list is not None:
            num_images = len(image_list)
            viz_list = [None for i in range(num_images)]
        else:
            raise KeyError("Nothing to show")

    f, axes = plt.subplots(1, num_images)
    for viz, image, annotations, ax in zip(viz_list, image_list, annotations_list, axes):
        if image is not None:
            ax.imshow(image, cmap="gray")
            if viz is not None:
                cax = ax.imshow(viz, alpha=0.3, cmap='jet')
            if annotations is not None:
                for a in annotations:
                    if len(a) == 4:
                        add_rect_annotations(a, ax, color="darkblue")
                    else:
                        add_mask_annotation(a, ax, colors="darkblue", contour_levels=[0.1])
        else:
            cax = ax.imshow(viz)
            if annotations is not None:
                for a in annotations:
                    if len(a) == 4:
                        add_rect_annotations(a, ax, color="darkblue")
                    else:
                        add_mask_annotation(a, ax, colors="darkblue", contour_levels=[0.1])

        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        f.subtitle(title)
    plt.show()


def add_rect_annotations(annotation, ax, color="darkblue"):
    x, y, width, height = annotation
    rect = plt.Rectangle((x, y), width, height, color=color, fill=False)
    ax.add_patch(rect)


def add_mask_annotation(annotation, ax, colors="darkblue", contour_levels=[0.5]):
    n, p = annotation.shape[0], annotation.shape[1]
    x, y = np.meshgrid(np.arange(0, n, step=1),
                       np.arange(0, p, step=1), indexing='ij')
    cax = ax.contour(y, x, annotation, contour_levels, colors=colors, linewidths=1)