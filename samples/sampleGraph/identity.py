import onnx
import numpy as np
from onnx import helper, TensorProto

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])

# W: [[1., 0.], [0., 1.]]
W_data = np.eye(2, dtype=np.float32)
W = helper.make_tensor(
    name='W',
    data_type=TensorProto.FLOAT,
    dims=W_data.shape,
    vals=W_data.flatten().tolist()
)

# y = x @ W
gemm_node = helper.make_node(
    op_type='Gemm',
    inputs=['x', 'W'],
    outputs=['y']
)

# create model
graph = helper.make_graph(
    nodes=[gemm_node],
    name='simple_gemm',
    inputs=[x],
    outputs=[y],
    initializer=[W]
)

model = helper.make_model(graph, producer_name='demo')
model.opset_import[0].version = 13

# save model
onnx.save(model, 'model.onnx')
print("model write to: model.onnx")


## verify
#import onnxruntime as ort
#import numpy as np
#sess = ort.InferenceSession('model.onnx')
#x_val = np.array([[3.0, 4.0]], dtype=np.float32)
#y_val = sess.run(None, {'x': x_val})[0]
#print(y_val)   # 输出 [[3. 4.]]
