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
2. Create a single project covering photos from all `dataset/raw_photos/<Category>/`
   folders (a combined export is what the import script now expects — see Task 3).
3. Import the photos from every `dataset/raw_photos/<Category>/` folder into that
   one project.
4. Labeling config: rectangle labels drawn tightly around each product, named
   with the item's full leaf-class path (e.g.
   `Ready-To-Eat/Instant-Noodles/Maggi/2-Minute-Curry`,
   `Snacks/Chocolate-Bar/Cadbury/Dairy-Milk`). Multiple boxes per photo are
   supported — the import script crops every box in every label file.
5. Export the finished annotations using Label Studio's **YOLO** export
   format. This produces an `images/` folder, a `labels/` folder (one `.txt`
   per image, one line per box as `class_id cx cy w h` normalized), and a
   `classes.txt` (class names indexed by line number).
6. Unzip/place the export directly here, merging into
   `dataset/raw_photos/{images,labels}/` and `dataset/raw_photos/classes.txt`
   (`images/` is gitignored — it's working-tree only, not committed).

## 3. Import

Run `python import_annotations.py` (see repo root) to crop every bounding box
in the combined export and route each crop into the destination directory
whose keyword matches the box's class name (substring match, e.g.
`"Instant-Noodles"` → `dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles/`,
`"Chocolate"` → `dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar/`).
Boxes whose class name doesn't match any configured keyword are skipped.
Add a new category by adding its raw-photo folder here and a new
`{keyword: dest_dir}` entry in `import_annotations.py`.
