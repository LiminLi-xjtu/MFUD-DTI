# 

SEED=3
EPOCHS=30
BATCH_SIZE=32

DATASET="BioSNAP"

# 定义任务列表，格式为 "SETTING|CONFIG_PATH"
TASKS=(
    "random|config_biosnap_r.json"
    "protein|config_biosnap_p.json"
    "scaffold|config_biosnap_s.json"
    "scaffold_protein|config_biosnap_sp.json"
)

for TASK in "${TASKS[@]}"; do
    # 解析 SETTING 和 CONFIG_PATH
    IFS='|' read -r SETTING CONFIG_PATH <<< "$TASK"
    
    echo "========================================"
    echo "开始运行数据集: $DATASET, 设置: $SETTING"
    echo "使用配置文件: $CONFIG_PATH"
    echo "========================================"

    for i in {1..5}; do
        echo "运行第 $i 次..."

        DATAFOLDER="./dataset/$DATASET/$SETTING"
        DATAFOLDER_PT="./dataset/$DATASET"
        # 结果路径包含 setting 和 config 名称以作区分
        RESULT_BASE_PATH="./result/$DATASET/${SETTING}"
        RESULT_PATH="${RESULT_BASE_PATH}_RUN_$i"

        mkdir -p "$RESULT_PATH"

        python main.py \
            --seed $SEED \
            --config_path "./config/biosnap/$CONFIG_PATH" \
            --datafolder "$DATAFOLDER" \
            --datafolder_pt "$DATAFOLDER_PT" \
            --result_path "$RESULT_PATH" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
    done
done