# #!/bin/bash
# 数据集列表
datasets=("cornell" "wisconsin" "texas")
lr=0.01
dropout=0.4
alpha=0.9
epochs=500
lr_gamma=0.0005
weight_decay=0.0005
hidden_size=512
output_size=512
task="node_classification"
str_aug="GB"
L=2
run=10

# 循环遍历数据集和学习率
for dataset in "${datasets[@]}"; do
    echo "正在运行数据集: $dataset"
    python  main.py --dataset "$dataset" \
                    --epochs "$epochs" \
                    --lr "$lr" \
                    --lr_gamma "$lr_gamma" \
                    --weight_decay "$weight_decay" \
                    --hidden_size "$hidden_size" \
                    --output_size "$output_size" \
                    --dropout "$dropout" \
                    --task "$task" \
                    --str_aug "$str_aug" \
                    --L "$L" \
                    --alpha "$alpha" \
                    --run "$run"
    echo "数据集 $dataset"
done
