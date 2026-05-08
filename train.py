from rfdetr import RFDETRNano

model = RFDETRNano()

# config:
dataset = "/media/elena/T7/BDD100K/coco"
epochs = 100
num_classes = 7
early_stopping = True
early_stopping_patience = 20
batch_size = 8
grad_accum_steps = 2 # tot BATCH SIZE = batch_size * grad_accum_steps
output_dir = "/path/to/output"

model.train(dataset_dir=dataset, run_test=False, epochs=epochs, num_classes=num_classes, early_stopping=early_stopping, early_stopping_patience=early_stopping_patience, batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=1e-4, output_dir=output_dir, freeze_encoder=False)