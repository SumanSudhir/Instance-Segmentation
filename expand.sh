HIERARCHY_FILE=challenge-2019-label300-segmentable-hierarchy.json
# BOUNDING_BOXES=challenge-2019-train-segmentation-bbox
# IMAGE_LABELS=challenge-2019-train-segmentation-labels
INSTANCE_SEGMENTATIONS=challenge-2019-validation-segmentation-masks


# python oid_hierarchical_labels_expansion.py \
#     --json_hierarchy_file=${HIERARCHY_FILE} \
#     --input_annotations=${BOUNDING_BOXES}.csv \
#     --output_annotations=${BOUNDING_BOXES}_expanded.csv \
#     --annotation_type=1

# python oid_hierarchical_labels_expansion.py \
#     --json_hierarchy_file=${HIERARCHY_FILE} \
#     --input_annotations=${IMAGE_LABELS}.csv \
#     --output_annotations=${IMAGE_LABELS}_expanded.csv \
#     --annotation_type=2

python oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${INSTANCE_SEGMENTATIONS}.csv \
    --output_annotations=${INSTANCE_SEGMENTATIONS}_expanded.csv \
    --annotation_type=1