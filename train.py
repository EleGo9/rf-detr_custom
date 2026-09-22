from rfdetr import RFDETRNano

model = RFDETRNano()
DATASET = "/media/elena/T9/HAura/mcap/lacisa_coco_format"

# config:
dataset = DATASET
epochs = 100
num_classes = 4
early_stopping = True
early_stopping_patience = 20
batch_size = 8
grad_accum_steps = 2 # tot BATCH SIZE = batch_size * grad_accum_steps
output_dir = "/home/elena/repos/rf-detr_custom/data/20260922_lacisa_coco_format"

model.train(dataset_dir=dataset, run_test=False, epochs=epochs, num_classes=num_classes, early_stopping=early_stopping, early_stopping_patience=early_stopping_patience, batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=1e-4, output_dir=output_dir, freeze_encoder=False)