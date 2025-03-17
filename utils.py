# 加载模型后运行此代码
def check_model_dtypes(model):
    print("模型权重数据类型分布:")
    dtype_counts = {}
    for name, param in model.named_parameters():
        dtype = param.dtype
        if dtype not in dtype_counts:
            dtype_counts[dtype] = 0
        dtype_counts[dtype] += 1
    
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 个参数")
    
    # 打印一些示例权重
    print("\n示例权重:")
    for name, param in list(model.named_parameters())[:3]:  # 前三个参数
        print(f"  {name}: 形状 {param.shape}, 类型 {param.dtype}")
        if param.numel() > 0:
            print(f"    值范围: [{param.min().item():.6f}, {param.max().item():.6f}]")
            if param.numel() > 5:
                print(f"    前5个值: {param.flatten()[:5]}")