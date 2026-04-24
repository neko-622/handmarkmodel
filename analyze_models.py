import onnx

# 分析ONNX模型
def analyze_onnx_model(model_path):
    print(f"\n=== 分析ONNX模型: {model_path} ===")
    model = onnx.load(model_path)
    
    # 获取输入信息
    print("\n输入信息:")
    for i, input in enumerate(model.graph.input):
        print(f"输入 {i}:")
        print(f"  名称: {input.name}")
        # 尝试获取形状信息
        if input.type.tensor_type.HasField('shape'):
            shape = input.type.tensor_type.shape
            shape_dim = []
            for dim in shape.dim:
                if dim.HasField('dim_value'):
                    shape_dim.append(dim.dim_value)
                else:
                    shape_dim.append('?')
            print(f"  形状: {shape_dim}")
        # 尝试获取数据类型
        data_type = input.type.tensor_type.elem_type
        print(f"  数据类型: {onnx.TensorProto.DataType.Name(data_type)}")
    
    # 获取输出信息
    print("\n输出信息:")
    for i, output in enumerate(model.graph.output):
        print(f"输出 {i}:")
        print(f"  名称: {output.name}")
        # 尝试获取形状信息
        if output.type.tensor_type.HasField('shape'):
            shape = output.type.tensor_type.shape
            shape_dim = []
            for dim in shape.dim:
                if dim.HasField('dim_value'):
                    shape_dim.append(dim.dim_value)
                else:
                    shape_dim.append('?')
            print(f"  形状: {shape_dim}")
        # 尝试获取数据类型
        data_type = output.type.tensor_type.elem_type
        print(f"  数据类型: {onnx.TensorProto.DataType.Name(data_type)}")

if __name__ == "__main__":
    # 分析ONNX模型
    onnx_model_path = "/workspace/ssne_ai_demo/cut_model.onnx"
    analyze_onnx_model(onnx_model_path)
