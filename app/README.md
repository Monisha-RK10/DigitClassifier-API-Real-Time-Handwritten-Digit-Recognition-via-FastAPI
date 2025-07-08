```bash
55,000 samples
↓  split into
859 batches (each batch = 64 samples)
↓
for each batch:
    → forward pass
    → compute loss
    → backprop
    → optimizer step
```
