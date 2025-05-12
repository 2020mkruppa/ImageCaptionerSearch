import dataset as DS
import model as Model
import trainer as Train
from types import SimpleNamespace
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

train_ds = DS.Dataset([0], DS.train_tfms)
val_ds = DS.Dataset([0], DS.valid_tfms)

model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768,
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)
train_config = SimpleNamespace(
    epochs = 5,
    freeze_epochs_gpt = 1,
    freeze_epochs_all = 2,
    lr = 1e-4,
    device = 'cuda',
    model_path = Path('captioner'),
    batch_size = 32
)

torch.manual_seed(0)
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=DS.collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=False,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=DS.collate_fn)
trainer = Train.Trainer(model_config,train_config,(train_dl,val_dl))

trainer.load_best_model()


def captionDirectory(imageDir, outfile, temp, repetitions, show):
	with open(outfile, "w") as file:
		num = 0
		for dirpath, dirnames, filenames in os.walk(imageDir):
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				gen_caption = ""
				for i in range(repetitions):			
					gen_caption += trainer.generate_caption(file_path, max_tokens=100, temperature=temp,deterministic=False) + " "
				gen_caption = gen_caption[:-1]
				if show:		
					plt.imshow(Image.open(file_path).convert('RGB'))		
					plt.title(f"model: {gen_caption}\ntemp: {temp}")
					plt.axis('off')
					plt.show()
				file.write(file_path + ":" + gen_caption + "\n")
				num += 1
				print(num)
				
    
captionDirectory(imageDir="Nature", outfile="captions.txt", temp=0.3, repetitions=2, show=False)
captionDirectory(imageDir="Photo_Portfolio", outfile="captions1.txt", temp=0.3, repetitions=2, show=False)
# https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch/output