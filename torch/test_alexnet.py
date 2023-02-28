import onnx
import onnxruntime as ort
import numpy as np


model = onnx.load('./model/alexnet.onnx')
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

ort_session = ort.InferenceSession('./model/alexnet.onnx')

for i in range(0, 100000):

    outputs = ort_session.run(None, {
        'input1': np.random.randn(1, 3, 224, 224).astype(np.float32)
    })

    print(outputs[0])
