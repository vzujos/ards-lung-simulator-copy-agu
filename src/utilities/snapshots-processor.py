# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:11:24 2026

@author: angus
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def crop_and_save_all(main_folder, left=0, right=0, top=0, bottom=0):
    """
    Batch crop all PNG images in RAW and save them into PROCESSED.

    Parameters
    ----------
    main_folder : str or Path
        Path to MAIN directory.
    left, right, top, bottom : int
        Pixels to remove from each side.
    """

    main = Path(main_folder)
    raw = main / "RAW"
    processed = main / "PROCESSED"

    processed.mkdir(exist_ok=True)

    for img_path in raw.glob("*.png"):

        img = Image.open(img_path)
        w, h = img.size

        cropped = img.crop((
            left,
            top,
            w - right,
            h - bottom
        ))

        new_name = img_path.stem + "-cropped.png"
        cropped.save(processed / new_name)

        print(f"Saved {new_name}")
        
        
def preview_crop(image_path, left=0, right=0, top=0, bottom=0):

    img = Image.open(image_path)
    w, h = img.size

    cropped = img.crop((
        left,
        top,
        w - right,
        h - bottom
    ))

    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    # Original image
    ax[0].imshow(img)
    ax[0].set_title("Original with crop limits")

    # Draw crop rectangle
    rect_x = [left, w-right, w-right, left, left]
    rect_y = [top, top, h-bottom, h-bottom, top]
    ax[0].plot(rect_x, rect_y)

    ax[0].axis("off")

    # Cropped result
    ax[1].imshow(cropped)
    ax[1].set_title("Cropped result")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    
tfol = "D:/2026-03-FigScreenshots/"    
    
preview_crop(
    tfol+"85a.png",
    left=0,
    right=0,
    top=420,
    bottom=550
)


crop_and_save_all(tfol,top=420,bottom=550)