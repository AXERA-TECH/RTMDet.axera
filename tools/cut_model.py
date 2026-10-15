import onnx

input_path = "./weights/yolo11s.onnx"
# output_path = "yolo11s-postprocss.onnx"
output_path = "./weights/yolo11s-backbone.onnx"
# input_names = ["/model.23/Concat_3_output_0"]
# output_names = ["output0"]

input_names = ["images"]
output_names = ["/model.23/Concat_1_output_0","/model.23/Concat_output_0","/model.23/Concat_2_output_0"]


onnx.utils.extract_model(input_path, output_path, input_names, output_names)