# Dataset JSONL and versioning

Export a dataset version to receive a ZIP containing `manifest.json`, `records.jsonl`, and an `images/` directory. Import a ZIP through Dataset versions. Keep the associated images in the archive: a JSONL path is relative to the exported bundle, never an arbitrary absolute path on the machine. A minimal manifest is:

```json
{"schema_version":1,"classes":["button","textbox","link","checkbox","dropdown","icon"],"source_version":"version-id","name":"My reviewed dataset","coordinate_format":"normalized_xyxy"}
```

Each line represents one instruction and one target (or no target). Multiple lines may reference the same screenshot. UTF-8 JSONL requires one complete JSON object per line.

```json
{"id":"example-search","image_id":"image-home","image_path":"images/home.png","width":1280,"height":720,"instruction":"Find the search field","target_present":true,"class_id":1,"bbox":[0.125,0.15,0.55,0.22],"click_point":[0.3375,0.185],"status":"reviewed","ambiguous":false,"group":"store-template-a","dataset_version":"version-id","split":"train"}
{"id":"example-absent","image_id":"image-home","image_path":"images/home.png","width":1280,"height":720,"instruction":"Find the checkout button","target_present":false,"class_id":null,"bbox":null,"click_point":null,"status":"reviewed","ambiguous":false,"group":"store-template-a","dataset_version":"version-id","split":"train"}
```

| Field | Meaning |
| --- | --- |
| `id` | Stable example identifier |
| `image_id`, `image_path` | Screenshot identity (1–100 ASCII letters, digits, underscores, or hyphens) and associated relative image path |
| `width`, `height` | Original decoded image dimensions |
| `instruction` | Nonempty target description |
| `target_present` | Boolean presence label |
| `class_id` | Zero-based index into manifest classes; null when absent |
| `bbox` | Normalized `[x_min,y_min,x_max,y_max]`; null when absent |
| `click_point` | Normalized `[x,y]` inside the target box; null when absent |
| `status` | `draft`, `reviewed`, or `excluded` |
| `ambiguous` | Review flag; flagged records cannot enter training snapshots |
| `group` | Website, template family, or collection session identifier |
| `dataset_version` | Source immutable snapshot identifier |
| `split` | `train`, `val`, or `test` in the exported snapshot |
| `synthetic` | Optional boolean identifying generated demonstration records |

Boxes satisfy `0 <= x_min < x_max <= 1` and `0 <= y_min < y_max <= 1`. Pixel coordinates are normalized x multiplied by original width and normalized y multiplied by original height. The frontend maintains normalized coordinates independently of display scaling, pan, and zoom. Click points can be adjusted manually; model predictions use a candidate point derived from the predicted box, rather than a separately supervised click head.

The default class list is `button`, `textbox`, `link`, `checkbox`, `dropdown`, `icon`. The manifest supplies the authoritative order. Configure classes before snapshot creation. Import and retraining require compatible class mappings; an index must retain its meaning.

## Review and validation

An uploaded image starts unannotated. Element annotations and instruction examples are distinct: an element may be referenced by many instructions. Saved examples remain drafts until explicitly reviewed. The image summary distinguishes unannotated, draft, reviewed, and excluded content. Missing instructions, invalid boxes, unknown classes, mismatched dimensions, and click points outside a box are errors.

Validation reports errors, warnings, review counts, class counts, absent-target counts, and split sizes. Only reviewed, nonambiguous, nonexcluded examples enter a new immutable snapshot. Adding or editing annotations later does not modify the snapshot or an existing run. Draft prediction corrections must be reviewed before versioning.

## Splits and immutability

Every instruction referring to one screenshot stays in the same split. Group-based splitting additionally keeps screenshots with the same nonempty group together. Choose a group broad enough to cover related pages; different filenames do not establish independence. Image-only grouping prevents shared-image leakage but does not infer website relationships.

The snapshot stores its split assignment, classes, annotations, and copied images with a fingerprint and image hashes. Training and resumption verify that snapshot content has not changed. The tokenizer is built using only training instructions; validation and test vocabulary does not influence fresh training.

A tiny collection or a single group may leave validation or test empty. Such results are not quality evidence. Use enough independent groups to reserve useful validation and test sets, and inspect the split statistics before training.

Import adds editable image/annotation records to the local library. Existing example IDs are rejected rather than overwritten; an existing image ID can receive additional distinct examples only if its image bytes and group agree. Image IDs remain stable metadata identifiers; generated physical filenames protect distinct IDs from Windows case folding and reserved device names. IDs outside the documented ASCII policy are rejected before writing images.

Create a new local snapshot after validation. Imported split labels and version identifiers describe the exported source; they do not silently attach the import to an existing immutable run. Import accepts at most 250 MB compressed, 500 MB expanded, and 10,000 entries; each image must be at most 40 MB and 40 megapixels. Duplicate ZIP member names, malformed JSON objects, invalid records, and unreadable images are rejected before the import commits annotations.
