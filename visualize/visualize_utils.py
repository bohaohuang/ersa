import matplotlib.pyplot as plt


def compare_two_figure(img_1, img_2, show_axis=False, fig_size=(12, 6)):
    """
    Show two figures in a row, link their axes
    :param img_1: image to show on the left
    :param img_2: image to show on the right
    :param show_axis: if False, axes will be hide
    :param fig_size: size of the figure
    :return:
    """
    plt.figure(figsize=fig_size)
    ax1 = plt.subplot(121)
    plt.imshow(img_1)
    if not show_axis:
        plt.axis('off')
    plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(img_2)
    if not show_axis:
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_three_figure(img_1, img_2, img_3, show_axis=False, fig_size=(12, 6)):
    """
    Show three figures in a row, link their axes
    :param img_1: image to show on the left
    :param img_2: image to show at the center
    :param img_3: image to show on the right
    :param show_axis: if False, axes will be hide
    :param fig_size: size of the figure
    :return:
    """
    plt.figure(figsize=fig_size)
    ax1 = plt.subplot(131)
    plt.imshow(img_1)
    if not show_axis:
        plt.axis('off')
    plt.subplot(132, sharex=ax1, sharey=ax1)
    plt.imshow(img_2)
    if not show_axis:
        plt.axis('off')
    plt.subplot(133, sharex=ax1, sharey=ax1)
    plt.imshow(img_3)
    if not show_axis:
        plt.axis('off')
    plt.tight_layout()
    plt.show()
