
input_file=${1:-"none"}
output_dir=${2:-""}
device=${3:-"0"}
sentence_model=${4:-"sentence-transformers/all-mpnet-base-v2"}

encoding_batch_size=65536
distance_distance_threshold=0.05
search_space_size=500
search_batch_size=1024
device=0

# Modify parameters
input_file="${DATA_MGT}/input_schema_example_difficulty_quality_category_language_safety.jsonl"
output_dir="${DATA_MGT}"

if [ $input_file == "none" ]; then
    echo "[magpie.sh] Input file not provided!"
    exit 1
fi
if [ ! -f $input_file ]; then
    echo "[magpie.sh] Input file not found!"
    exit 1
fi

# get job path from input file
job_path=$(dirname "$input_file")
exec > >(tee -a "$job_path/remove_repetition.log") 2>&1
echo "[magpie.sh] Job Path: $job_path"
echo "[magpie.sh] Input File: $input_file"
echo "[magpie.sh] Output Directory: $output_dir"
echo "[magpie.sh] Model Name: $sentence_model"
echo "[magpie.sh] System Config: device=$device, encoding_batch_size=$encoding_batch_size, distance_distance_threshold=$distance_distance_threshold, search_space_size=$search_space_size, search_batch_size=$search_batch_size"

echo "[magpie.sh] Start Removing Repetitions..."
CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/gen_dis.py \
    --device $device \
    --sentence_model $sentence_model \
    --input_file $input_file \
    --encoding_batch_size $encoding_batch_size \
    --distance_distance_threshold $distance_distance_threshold \
    --search_space_size $search_space_size \
    --search_batch_size $search_batch_size \
    --output_dir $output_dir \

echo "[magpie.sh] Finish Removing Repetitions"