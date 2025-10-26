# Diversity test for long, short prompts
./script/gen_image.sh \
    --dataset_type long \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/long_prompt

./script/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt