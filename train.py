from rfdetr import RFDETRNano

DATASET = "/media/elena/T9/HAura/mcap/lacisa_coco_format"

# config:
dataset = DATASET
epochs = 100
num_classes = 4
freeze_encoder = False
early_stopping = True
early_stopping_patience = 20
batch_size = 8
grad_accum_steps = 2 # tot BATCH SIZE = batch_size * grad_accum_steps
output_dir = "/home/elena/repos/rf-detr_custom/data/20260922_lacisa_coco_format"

# num_classes/freeze_encoder are ModelConfig fields (rfdetr >= 1.8): they belong
# on the constructor, not on .train() — TrainConfig rejects unknown kwargs since 1.8.
model = RFDETRNano(num_classes=num_classes, freeze_encoder=freeze_encoder)

model.train(dataset_dir=dataset, run_test=False, epochs=epochs, early_stopping=early_stopping, early_stopping_patience=early_stopping_patience, batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=1e-4, output_dir=output_dir)