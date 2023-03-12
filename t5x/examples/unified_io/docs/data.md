# Tasks, Preprocessors, and Mixtures

## Tasks
To add a task, register the task in `t5x/examples/unified_io/data/tasks.py`

This would look something like:
```python
TaskRegistry.add(
    "detection_coco_2017",
    source=seqio.TfdsDataSource(
        tfds_name="coco_all:1.0.1",
        tfds_data_dir=TFDS_DATA_DIR,
        ),
    preprocessors=[
        functools.partial(
            rekey, key_map={
                "image/filename": ["image/filename"],
                "image": ["image"],
                "bbox": ["objects", "bbox"],
                "label": ["objects", "label"],
            }),
        functools.partial(
            detection_preprocessor,
            class_name = load_class_name('metadata/coco/coco_class_name_2017.json'),
            detect_all_instances = True,
        ),
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    postprocess_fn=get_id,
    metric_fns=[
        functools.partial(save_grit_predictions, task='detection')
    ],
    output_features=FINETUNE_OUTPUT_FEATURES,
)
```

* `source`: specifies the raw source - generally a tfds
* `preprocessors`: a list of preprocessing functions to be applied in sequence each of which takes in the previous dataset and creates a new dataset
* `postprocess_fn`: specifies how to post process the gt and target predictions prior to computing metrics
* `metric_fns`: metrics to compute
* `output_features`: a dictionary specifying names and types of output features

## Preprocessing
The preprocessors are specified in `t5x/examples/unified_io/data/preprocessors.py`
Here's a list of commonly used preprocessor:
- key mapping
    * `rekey`: replaces features keys according to a provided key map
- pretraining tasks (masked image/language modeling)
    * `multimodal_prefix_preprocessor`: add image/text masks (paired image-text tasks)
    * `image_prefix_preprocessor`: add image masks (image only tasks)
    * `text_prefix_preprocessor`: add text masks (text only tasks)
- downstream tasks
    * `image_tagging_preprocessor`
    * `detection_preprocessor`
    * `class_specific_detection_preprocessor`
    * `box_classification_preprocessor`
    * `semantic_segmentation_preprocessor`
    * `image_caption_preprocessor`
    * `image_generation_preprocessor`
    * `image_inpainting_preprocessor`
    * `segmentation_based_image_generation_preprocessor`
    * `vqa_preprocessor`
    * `refexp_preprocessor`
    * `rel_predict_preprocessor`
    * `depth_estimation_preprocessor`
    * `pose_estimation_preprocessor`
    * `region_caption_preprocessor`
- dataset specific
    * `vizwiz_vqa_preprocessor`
    * `vcr_preprocessor`
    * `viscommet_preprocessor`

## How to verify your dataset works?

Here's a testing script that loads and iterates through the dataset of interest: `t5x/examples/unified_io/dataset_test.py`

Change the name of the dataset being fed to `seqio.get_mixture_or_task()` and launch the script. You may put `pdb` or `ipdb` breakpoints in the preprocessing code to check value of variables.
