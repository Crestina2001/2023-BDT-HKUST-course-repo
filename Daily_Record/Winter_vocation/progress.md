## 12.22

Try to change the CNN on sequential images into CNN+self-attention, but it does not work. Actually, the original code may work well, because the kernel layer can actually obtain information from neighboring images efficiently:

```python
Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3)
```

