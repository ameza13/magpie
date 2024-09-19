# Application starts here
input_file=${1:-"none"}
tag_mission=${2:-"all"}
device=${3:-"0"}
model_path=${4:-"mistralai/Mistral-7B-Instruct-v0.3"}
guard_model_path="allenai/wildguard"
reward_model_path="sfairXC/FsfairX-LLaMA3-RM-v0.1" # NOT OPEN SOURCE
tensor_parallel=8
gpu_memory_utilization=0.95
batch_size=1000

device="0,1,2,3,4,5,6,7"

# Modify parameters
input_file="${DATA_MGT}/input_schema_example.jsonl"
output_dir="${DATA_MGT}"

if [ $input_file == "none" ]; then
    echo "[magpie.sh] Input file not provided!"
    exit 1
fi
if [ ! -f $input_file ]; then
    echo "[magpie.sh] Input file not found!"
    exit 1
fi

# tag_mission="reward" # Will only run reward

# get job path from input file
job_path=$(dirname "$input_file")
exec > >(tee -a "$job_path/tagging.log") 2>&1
echo "[magpie.sh] Job Path: $job_path"
echo "[magpie.sh] Input File: $input_file"
echo "[magpie.sh] Output Directory: $output_dir"
echo "[magpie.sh] Tagging Mission: $tag_mission"
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] System Config: device=$device, batch_size=$batch_size, tensor_parallel=$tensor_parallel"

if [ $tag_mission == "difficulty" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Difficulty Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "difficulty" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --output_dir $output_dir \

    echo "[magpie.sh] Finish Generating Difficulty Tags!"

    # Change input file name to difficulty tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    difficulty_tag_file="${input_file_dir}/${input_file_name_no_ext}_difficulty.${input_file_ext}"
    input_file=$difficulty_tag_file
    echo "[magpie.sh] Difficulty Tagged File: $input_file"
fi

if [ $tag_mission == "quality" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Quality Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "quality" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --output_dir $output_dir \

    echo "[magpie.sh] Finish Generating Quality Tags!"

    # Change input file name to quality tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_quality.${input_file_ext}"
    input_file=$quality_tag_file
    echo "[magpie.sh] Quality Tagged File: $input_file"
fi

if [ $tag_mission == "classification" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Task Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "classification" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --output_dir $output_dir \

    echo "[magpie.sh] Finish Generating Task Tags!"

    # Change input file name to task tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    task_tag_file="${input_file_dir}/${input_file_name_no_ext}_category.${input_file_ext}"
    input_file=$task_tag_file
    echo "[magpie.sh] Task Tagged File: $input_file"
fi

if [ $tag_mission == "language" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Language Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --input_file $input_file \
        --tag_mission "language" \
        --output_dir $output_dir \

    echo "[magpie.sh] Finish Generating Language Tags!"

    # Change input file name to quality tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    language_tag_file="${input_file_dir}/${input_file_name_no_ext}_language.${input_file_ext}"
    input_file=$language_tag_file
    echo "[magpie.sh] Language Tagged File: $input_file"
fi

if [ $tag_mission == "sample_quality" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Quality Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "sample_quality" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --output_dir $output_dir \

    echo "[magpie.sh] Finish Generating Quality Tags!"

    # Change input file name to sample quality tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    sample_quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_sample-quality.${input_file_ext}"
    input_file=$sample_quality_tag_file
    echo "[magpie.sh] Quality Tagged File: $input_file"
fi

# if [ $tag_mission == "safety" ] || [ $tag_mission == "all" ]; then
#     echo "[magpie.sh] Start Generating Safety Tags..."
#     CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
#         --device $device \
#         --guard_model_path $guard_model_path \
#         --input_file $input_file \
#         --tag_mission "safety" \
#         --tensor_parallel $tensor_parallel \
#         --gpu_memory_utilization $gpu_memory_utilization \
#         --batch_size $batch_size \
#         --output_dir $output_dir \

#     echo "[magpie.sh] Finish Generating Safety Tags!"

#     # Change input file name to quality tagged file
#     input_file_name=$(basename $input_file)
#     input_file_dir=$(dirname $input_file)
#     input_file_name_no_ext="${input_file_name%.*}"
#     input_file_ext="${input_file_name##*.}"
#     safety_tag_file="${input_file_dir}/${input_file_name_no_ext}_safety.${input_file_ext}"
#     input_file=$safety_tag_file
#     echo "[magpie.sh] Safety Tagged File: $input_file"
# fi

# TO DO: Define what reward model we should use
# if [ $tag_mission == "reward" ] || [ $tag_mission == "all" ]; then
#     echo "[magpie.sh] Start Generating Reward Tags..."
#     python ${WORKSPACE}/magpie/exp/unitag.py \
#         --device $device \
#         --reward_model_path $reward_model_path \
#         --input_file $input_file \
#         --tag_mission "reward" \
#         --tensor_parallel $tensor_parallel \
#         --batch_size 1 \

#     echo "[magpie.sh] Finish Generating Reward Tags!"

#     # Change input file name to quality tagged file
#     input_file_name=$(basename $input_file)
#     input_file_dir=$(dirname $input_file)
#     input_file_name_no_ext="${input_file_name%.*}"
#     input_file_ext="${input_file_name##*.}"
#     reward_tag_file="${input_file_dir}/${input_file_name_no_ext}_reward.${input_file_ext}"
#     input_file=$reward_tag_file
#     echo "[magpie.sh] Reward Tagged File: $input_file"
# fi

if [ $tag_mission == "conversation_quality" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Quality Tags..."
    CUDA_VISIBLE_DEVICES=$device python ${WORKSPACE}/magpie/exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "conversation_quality" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --max_tokens $max_tokens \
        --max_model_len $max_model_len

    echo "[magpie.sh] Finish Generating Quality Tags!"

    # Change input file name to quality tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_conversation_quality.${input_file_ext}"
    input_file=$quality_tag_file
    echo "[magpie.sh] Quality Tagged File: $input_file"
fi

echo "[magpie.sh] Finish Tagging Mission: $tag_mission"