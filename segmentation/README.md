# Manual lesion-mask prototype

The downloaded BreaKHis 400X data contains image-level `benign`/`malignant`
labels, but no pixel-level lesion masks. This directory defines a small,
manually annotated prototype without fabricating medical annotations.

## Annotation workflow

1. Select a small, representative sample from both classes.
2. Copy the source images into `segmentation/images/`.
3. Open them in CVAT, Label Studio, or another approved annotation tool.
4. Draw a binary polygon or brush mask around the visible lesion/tumor region.
5. Export each mask as an 8-bit PNG with the exact source stem:
   `images/SOB_xxx.png` -> `masks/SOB_xxx.png`.
6. Use only black (`0`) for background and white (`255`) for lesion.
7. Have a second reviewer check the annotations before training.
8. Run the manifest validator:

```powershell
python segmentation\validate_masks.py `
  --images segmentation\images `
  --masks segmentation\masks
```

Do not treat Grad-CAM heatmaps as segmentation ground truth. They are model
explanations and do not replace expert pixel annotations.

## Required layout

```text
segmentation/
  images/
    sample_001.png
  masks/
    sample_001.png
```

The prototype should remain separate from the classification train/test split
until patient-level groups and annotation-review status have been recorded.
