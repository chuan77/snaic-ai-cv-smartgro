# Raw photo staging + Label Studio annotation

This directory holds product photos *before* they're cropped and merged into
`dataset/GroceryStoreDataset/dataset/train/`. It is intentionally outside the
`GroceryStoreDataset/` folder, which is its own git repo.

## 1. Drop in raw photos

- `dataset/raw_photos/Instant-Noodles/` — one photo per instant-noodle item,
  single product per photo, any background.
- `dataset/raw_photos/Chocolate-Bar/` — one photo per chocolate-bar item, same rule.

## 2. Annotate in Label Studio

1. Install/run Label Studio (`pip install label-studio && label-studio start`,
   or your existing instance).
2. Create one project per category (or one project with two labels — either
   works, since each category is imported separately in Task 4).
3. Import the photos from the matching `dataset/raw_photos/<Category>/` folder.
4. Labeling config: a single rectangle label per photo, drawn tightly around
   the product. Exactly one bounding box per image — the import script in
   this repo only reads the first box in each label file.
5. Export the finished annotations using Label Studio's **YOLO** export
   format. This produces an `images/` folder, a `labels/` folder (one `.txt`
   per image, `class_id cx cy w h` normalized), and a `classes.txt`.
6. Unzip/place the export here:
   - `dataset/raw_photos/Instant-Noodles/labelstudio_export/{images,labels}/`
   - `dataset/raw_photos/Chocolate-Bar/labelstudio_export/{images,labels}/`

## 3. Import

Run `python import_annotations.py` (see repo root) to crop each annotated
photo to its bounding box and copy it into
`dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles/` and
`dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar/` respectively.
